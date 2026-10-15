//! Rekurrente Schichten für Folgen: [`Rnn`], [`Gru`] und [`Lstm`].
//!
//! # Eine Folge ist eine Operation
//!
//! Alle drei verarbeiten **eine ganze Folge fester Länge** `STEPS` auf einmal. Die Eingabe sind
//! `STEPS` Zeilen zu je `IN` Werten (`input[zeitschritt][merkmal]`), die Ausgabe `STEPS` Zeilen
//! zu je `H` Werten: der "versteckte Zustand" `h` nach jedem Schritt. Weil die Folgenlänge im Typ
//! steht, geht das ohne `alloc`. Variable Längen gibt es ohne Heap nur mit Auffüllen (Padding).
//!
//! Die Schicht hat ein Gedächtnis: In jedem Schritt geht der Zustand des vorigen Schritts mit
//! ein, und der **gleiche** Satz Gewichte gilt in allen Schritten.
//!
//! # Rückwärts über die Zeit
//!
//! Beim Zurückrechnen (BPTT, "Backpropagation Through Time") läuft der Gradient die Schritte
//! rückwärts ab. Ein Schritt bekommt dabei zwei Gradienten: den von der Ausgabe und den vom
//! **nächsten Schritt**, der ja den Zustand benutzt hat. Alles steckt in `backward`. Die
//! Zwischenwerte der Tore werden dort aus der Eingabe und der bekannten Ausgabe (den Zuständen)
//! neu berechnet, damit der `Block` dafür keinen zusätzlichen Speicher braucht. Das kostet
//! etwa eine zweite Vorwärtsrechnung.
//!
//! Aus der Folge eine einzige Zahlengruppe zu machen (Klassifikation einer ganzen Folge), erledigt
//! [`LastStep`](crate::sequence::LastStep).
//!
//! # Welche?
//!
//! | | Tore | Parameter (in `H`, `IN`) | gut für |
//! |---|---|---|---|
//! | [`Rnn`] | keins | `H·(IN + H + 1)` | kurze Folgen, Einstieg |
//! | [`Gru`] | 2 + Kandidat | `3·H·(IN + H + 1)` | längere Folgen, wenig Aufwand |
//! | [`Lstm`] | 3 + Kandidat, extra Zellzustand | `4·H·(IN + H + 1)` | lange Abhängigkeiten |
//!
//! Das einfache RNN vergisst schnell, weil der Gradient bei jedem Schritt mit denselben Gewichten
//! multipliziert wird und dadurch nach vielen Schritten verschwindet oder explodiert. Die Tore von
//! GRU und LSTM lassen den Zustand fast unverändert durchlaufen, wenn es nötig ist. Gegen
//! explodierende Gradienten hilft zusätzlich `ClipNorm` und eine orthogonale Initialisierung
//! (`Orthogonal`).
//!
//! `Rnn` braucht nur eine Aktivierung (`Tanh` oder, ohne Mathe-Bibliothek, `HardTanh`). `Gru` und
//! `Lstm` brauchen `exp` und `tanh`, deshalb gibt es sie nur mit dem Feature `std` oder `libm`.

use core::marker::PhantomData;

use crate::activation::Activation;
use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;

/// Die Gewichte eines Tors: `((Wx, Wh), b)`. `Wx` (`H × IN`) gewichtet die Eingabe des Schritts,
/// `Wh` (`H × H`) den Zustand des vorigen Schritts, `b` ist der Bias. Ein Tor berechnet
/// `a = Wx·x + Wh·h + b`.
pub type Gate<const IN: usize, const H: usize, T = f32> = (([[T; IN]; H], [[T; H]; H]), [T; H]);

/// Startwerte eines Tors: `Wx` und `Wh` aus dem `Initializer`, Bias 0.
fn init_gate<const IN: usize, const H: usize, T: Scalar, I: Initializer>(
    initializer: &I,
    rng: &mut Rng,
) -> Gate<IN, H, T> {
    let (mut wx, mut wh) = ([[T::ZERO; IN]; H], [[T::ZERO; H]; H]);
    initializer.fill(rng, &mut wx, IN, H);
    initializer.fill(rng, &mut wh, H, H);
    ((wx, wh), [T::ZERO; H])
}

fn dot<T: Scalar, const N: usize>(a: &[T; N], b: &[T; N]) -> T {
    a.iter().zip(b).fold(T::ZERO, |sum, (&x, &y)| sum + x * y)
}

/// `a = Wx·x + Wh·h + b`
fn gate_forward<const IN: usize, const H: usize, T: Scalar>(
    ((wx, wh), bias): &Gate<IN, H, T>,
    x: &[T; IN],
    h: &[T; H],
) -> [T; H] {
    core::array::from_fn(|j| bias[j] + dot(&wx[j], x) + dot(&wh[j], h))
}

/// Rückwärts zu [`gate_forward`]: Mit dem Gradienten `da` nach `a` **addiert** es zu den
/// Gradienten des Tors und zu `dx` (nach der Eingabe) und `dh` (nach dem Zustand):
/// `∂L/∂Wx[j][i] += da[j]·x[i]`, `∂L/∂Wh[j][k] += da[j]·h[k]`, `∂L/∂b[j] += da[j]`,
/// `∂L/∂x[i] += Σ_j Wx[j][i]·da[j]`, `∂L/∂h[k] += Σ_j Wh[j][k]·da[j]`.
fn gate_backward<const IN: usize, const H: usize, T: Scalar>(
    ((wx, wh), _): &Gate<IN, H, T>,
    ((grad_wx, grad_wh), grad_bias): &mut Gate<IN, H, T>,
    x: &[T; IN],
    h: &[T; H],
    da: &[T; H],
    dx: &mut [T; IN],
    dh: &mut [T; H],
) {
    for (j, &g) in da.iter().enumerate() {
        grad_bias[j] += g;
        for (i, &xi) in x.iter().enumerate() {
            grad_wx[j][i] += g * xi;
            dx[i] += wx[j][i] * g;
        }
        for (k, &hk) in h.iter().enumerate() {
            grad_wh[j][k] += g * hk;
            dh[k] += wh[j][k] * g;
        }
    }
}

/// Einfaches rekurrentes Netz (Elman): `h[t] = φ(Wx·x[t] + Wh·h[t−1] + b)` mit `h[−1] = 0`.
///
/// Die Aktivierung `φ` ist wählbar: `Tanh` ist der Standard, `HardTanh` kommt ohne Mathe-Bibliothek
/// aus. Die Parameter sind ein einziges [`Gate`].
///
/// ```
/// use neuronws_layers::activation::HardTanh;
/// use neuronws_layers::recurrent::{Rnn, RnnLayer};
/// use neuronws_layers::traits::ForwardPath;
///
/// // 3 Schritte, 1 Merkmal, 1 Zustand. h[t] = clamp(1.0·x[t] + 0.5·h[t−1])
/// let layer = RnnLayer::<3, 1, 1, HardTanh>::from_parts(
///     Rnn::new(HardTanh),
///     (([[1.0]], [[0.5]]), [0.0]),
/// );
/// let h = layer.forward(&[[0.5], [0.5], [0.5]]);
/// assert_eq!(h, [[0.5], [0.75], [0.875]]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Rnn<const STEPS: usize, const IN: usize, const H: usize, A: Activation, T: Scalar = f32>
{
    activation: A,
    scalar: PhantomData<T>,
}

impl<const STEPS: usize, const IN: usize, const H: usize, A: Activation, T: Scalar>
    Rnn<STEPS, IN, H, A, T>
{
    pub const fn new(activation: A) -> Self {
        Self {
            activation,
            scalar: PhantomData,
        }
    }
}

impl<const STEPS: usize, const IN: usize, const H: usize, A: Activation, T: Scalar> Operation
    for Rnn<STEPS, IN, H, A, T>
{
    type Scalar = T;
    type Input = [[T; IN]; STEPS];
    type Output = [[T; H]; STEPS];
    type Params = Gate<IN, H, T>;

    fn forward(&self, gate: &Self::Params, input: &Self::Input) -> Self::Output {
        let mut output = [[T::ZERO; H]; STEPS];
        let mut h = [T::ZERO; H];
        for (x, out) in input.iter().zip(output.iter_mut()) {
            h = gate_forward(gate, x, &h).map(|z| self.activation.apply(z));
            *out = h;
        }
        output
    }

    /// Rückwärts über die Zeit. Im Schritt `t` ist der Gradient nach dem Zustand
    /// `dh = g[t] + (Gradient vom Schritt t+1)`. Daraus folgt der Gradient vor der Aktivierung
    /// `da = dh · φ'`, den `gate_backward` auf Gewichte, Eingabe und den vorigen Zustand verteilt.
    fn backward(
        &self,
        gate: &Self::Params,
        grad_gate: &mut Self::Params,
        input: &Self::Input,
        output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let mut grad_input = [[T::ZERO; IN]; STEPS];
        let mut dh_next = [T::ZERO; H];
        for t in (0..STEPS).rev() {
            let h_prev = if t > 0 { output[t - 1] } else { [T::ZERO; H] };
            // Die Ableitung braucht manchmal die Eingabe `z` der Aktivierung: neu berechnen.
            let z = gate_forward(gate, &input[t], &h_prev);
            let da: [T; H] = core::array::from_fn(|j| {
                (grad_output[t][j] + dh_next[j])
                    * self.activation.derivative_with_output(z[j], output[t][j])
            });
            let (mut dx, mut dh_prev) = ([T::ZERO; IN], [T::ZERO; H]);
            gate_backward(
                gate,
                grad_gate,
                &input[t],
                &h_prev,
                &da,
                &mut dx,
                &mut dh_prev,
            );
            grad_input[t] = dx;
            dh_next = dh_prev;
        }
        grad_input
    }

    /// `Wx` und `Wh` aus dem `Initializer`, Bias 0. Für `Wh` ist `Orthogonal` die klassische Wahl.
    fn init_params<I: Initializer>(&self, initializer: &I, rng: &mut Rng) -> Self::Params {
        init_gate(initializer, rng)
    }
}

/// Eine RNN-Schicht.
pub type RnnLayer<const STEPS: usize, const IN: usize, const H: usize, A, T = f32> =
    Block<Rnn<STEPS, IN, H, A, T>>;

/// Sigmoid, gebraucht für die Tore von GRU und LSTM.
#[cfg(any(feature = "std", feature = "libm"))]
fn sigmoid<T: Scalar>(z: T) -> T {
    T::ONE / (T::ONE + (-z).exp())
}

/// GRU (Gated Recurrent Unit): ein rekurrentes Netz mit zwei Toren, die steuern, wie viel vom
/// alten Zustand bleibt.
///
/// ```text
/// r = σ(Wr·x + Ur·h + br)               Rücksetz-Tor: wie viel vom alten Zustand fließt in den Kandidaten
/// z = σ(Wz·x + Uz·h + bz)               Update-Tor:   wie viel vom alten Zustand bleibt
/// n = tanh(Wn·x + Un·(r ⊙ h) + bn)      Kandidat
/// h' = (1 − z) ⊙ n + z ⊙ h
/// ```
///
/// Ist `z` nahe 1, wird der Zustand einfach durchgereicht, und der Gradient fließt fast
/// ungedämpft zurück. So behält die GRU Information über viele Schritte. Die Parameter sind
/// drei [`Gate`]s in der Reihenfolge `((r, z), n)`.
///
/// Braucht `exp` und `tanh`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Gru<const STEPS: usize, const IN: usize, const H: usize, T: Scalar = f32>(
    PhantomData<T>,
);

#[cfg(any(feature = "std", feature = "libm"))]
impl<const STEPS: usize, const IN: usize, const H: usize, T: Scalar> Gru<STEPS, IN, H, T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    /// Die drei Tor-Ergebnisse `(r, z, n)` eines Schritts aus Eingabe `x` und Zustand `h`.
    #[allow(clippy::type_complexity)]
    fn gates(
        ((reset, update), candidate): &<Self as Operation>::Params,
        x: &[T; IN],
        h: &[T; H],
    ) -> ([T; H], [T; H], [T; H]) {
        let r = gate_forward(reset, x, h).map(sigmoid);
        let z = gate_forward(update, x, h).map(sigmoid);
        let reset_h: [T; H] = core::array::from_fn(|j| r[j] * h[j]);
        let n = gate_forward(candidate, x, &reset_h).map(Scalar::tanh);
        (r, z, n)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<const STEPS: usize, const IN: usize, const H: usize, T: Scalar> Operation
    for Gru<STEPS, IN, H, T>
{
    type Scalar = T;
    type Input = [[T; IN]; STEPS];
    type Output = [[T; H]; STEPS];
    type Params = ((Gate<IN, H, T>, Gate<IN, H, T>), Gate<IN, H, T>);

    fn forward(&self, params: &Self::Params, input: &Self::Input) -> Self::Output {
        let mut output = [[T::ZERO; H]; STEPS];
        let mut h = [T::ZERO; H];
        for (x, out) in input.iter().zip(output.iter_mut()) {
            let (_, z, n) = Self::gates(params, x, &h);
            h = core::array::from_fn(|j| (T::ONE - z[j]) * n[j] + z[j] * h[j]);
            *out = h;
        }
        output
    }

    /// Mit `dh` (Gradient nach dem Zustand, inklusive dem vom nächsten Schritt):
    ///
    /// ```text
    /// dn = dh ⊙ (1 − z)        dz = dh ⊙ (h_alt − n)        dh_alt = dh ⊙ z
    /// d(an) = dn ⊙ (1 − n²)    d(r ⊙ h_alt) = Unᵀ · d(an)
    /// dr = d(r ⊙ h_alt) ⊙ h_alt        dh_alt += d(r ⊙ h_alt) ⊙ r
    /// d(ar) = dr ⊙ r (1 − r)           d(az) = dz ⊙ z (1 − z)
    /// ```
    ///
    /// Die drei `d(a…)` gehen durch `gate_backward` auf die Gewichte, die Eingabe und `h_alt`.
    fn backward(
        &self,
        params: &Self::Params,
        grads: &mut Self::Params,
        input: &Self::Input,
        output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let ((reset, update), candidate) = params;
        let ((grad_reset, grad_update), grad_candidate) = grads;
        let mut grad_input = [[T::ZERO; IN]; STEPS];
        let mut dh_next = [T::ZERO; H];

        for t in (0..STEPS).rev() {
            let h_prev = if t > 0 { output[t - 1] } else { [T::ZERO; H] };
            let x = &input[t];
            let (r, z, n) = Self::gates(params, x, &h_prev);
            let dh: [T; H] = core::array::from_fn(|j| grad_output[t][j] + dh_next[j]);

            let mut dx = [T::ZERO; IN];
            let mut dh_prev: [T; H] = core::array::from_fn(|j| dh[j] * z[j]);

            // Kandidat: n = tanh(an), an = Wn·x + Un·(r ⊙ h_alt) + bn
            let da_n: [T; H] =
                core::array::from_fn(|j| dh[j] * (T::ONE - z[j]) * (T::ONE - n[j] * n[j]));
            let reset_h: [T; H] = core::array::from_fn(|j| r[j] * h_prev[j]);
            let mut d_reset_h = [T::ZERO; H];
            gate_backward(
                candidate,
                grad_candidate,
                x,
                &reset_h,
                &da_n,
                &mut dx,
                &mut d_reset_h,
            );

            // Rücksetz-Tor: r beeinflusst den Kandidaten über r ⊙ h_alt
            let da_r: [T; H] =
                core::array::from_fn(|j| d_reset_h[j] * h_prev[j] * r[j] * (T::ONE - r[j]));
            for j in 0..H {
                dh_prev[j] += d_reset_h[j] * r[j];
            }
            gate_backward(reset, grad_reset, x, &h_prev, &da_r, &mut dx, &mut dh_prev);

            // Update-Tor: z mischt n und h_alt
            let da_z: [T; H] =
                core::array::from_fn(|j| dh[j] * (h_prev[j] - n[j]) * z[j] * (T::ONE - z[j]));
            gate_backward(
                update,
                grad_update,
                x,
                &h_prev,
                &da_z,
                &mut dx,
                &mut dh_prev,
            );

            grad_input[t] = dx;
            dh_next = dh_prev;
        }
        grad_input
    }

    /// `Wx` und `Wh` aller Tore aus dem `Initializer`, alle Biase 0.
    fn init_params<I: Initializer>(&self, initializer: &I, rng: &mut Rng) -> Self::Params {
        (
            (init_gate(initializer, rng), init_gate(initializer, rng)),
            init_gate(initializer, rng),
        )
    }
}

/// Eine GRU-Schicht.
#[cfg(any(feature = "std", feature = "libm"))]
pub type GruLayer<const STEPS: usize, const IN: usize, const H: usize, T = f32> =
    Block<Gru<STEPS, IN, H, T>>;

/// LSTM (Long Short-Term Memory): ein rekurrentes Netz mit **drei Toren und einem eigenen
/// Zellzustand** `c`, der fast ungestört durch die Zeit läuft.
///
/// ```text
/// i = σ(Wi·x + Ui·h + bi)          Eingabe-Tor:  wie viel Neues in die Zelle geschrieben wird
/// f = σ(Wf·x + Uf·h + bf)          Vergiss-Tor:  wie viel von der alten Zelle bleibt
/// g = tanh(Wg·x + Ug·h + bg)       Kandidat
/// o = σ(Wo·x + Uo·h + bo)          Ausgabe-Tor:  wie viel von der Zelle sichtbar wird
/// c' = f ⊙ c + i ⊙ g
/// h' = o ⊙ tanh(c')
/// ```
///
/// Die Zelle `c` wird nur multipliziert und addiert, ohne dass ein Gewicht dazwischen steht.
/// Deshalb kann der Gradient über viele Schritte zurückfließen. Der Bias des Vergiss-Tors startet
/// bei 1 (alle anderen bei 0): Das Netz erinnert sich anfangs lieber, als dass es vergisst, was
/// das Lernen langer Abhängigkeiten deutlich erleichtert. Die Parameter sind vier [`Gate`]s in
/// der Reihenfolge `((i, f), (g, o))`. Ausgegeben wird der Zustand `h` jedes Schrittes.
///
/// Braucht `exp` und `tanh`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Lstm<const STEPS: usize, const IN: usize, const H: usize, T: Scalar = f32>(
    PhantomData<T>,
);

/// Die vier Tor-Ergebnisse `(i, f, g, o)` eines LSTM-Schritts.
#[cfg(any(feature = "std", feature = "libm"))]
type LstmGates<const H: usize, T> = ([T; H], [T; H], [T; H], [T; H]);

#[cfg(any(feature = "std", feature = "libm"))]
impl<const STEPS: usize, const IN: usize, const H: usize, T: Scalar> Lstm<STEPS, IN, H, T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    fn gates(
        ((input_gate, forget_gate), (candidate, output_gate)): &<Self as Operation>::Params,
        x: &[T; IN],
        h: &[T; H],
    ) -> LstmGates<H, T> {
        (
            gate_forward(input_gate, x, h).map(sigmoid),
            gate_forward(forget_gate, x, h).map(sigmoid),
            gate_forward(candidate, x, h).map(Scalar::tanh),
            gate_forward(output_gate, x, h).map(sigmoid),
        )
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<const STEPS: usize, const IN: usize, const H: usize, T: Scalar> Operation
    for Lstm<STEPS, IN, H, T>
{
    type Scalar = T;
    type Input = [[T; IN]; STEPS];
    type Output = [[T; H]; STEPS];
    type Params = (
        (Gate<IN, H, T>, Gate<IN, H, T>),
        (Gate<IN, H, T>, Gate<IN, H, T>),
    );

    fn forward(&self, params: &Self::Params, input: &Self::Input) -> Self::Output {
        let mut output = [[T::ZERO; H]; STEPS];
        let (mut h, mut c) = ([T::ZERO; H], [T::ZERO; H]);
        for (x, out) in input.iter().zip(output.iter_mut()) {
            let (i, f, g, o) = Self::gates(params, x, &h);
            c = core::array::from_fn(|j| f[j] * c[j] + i[j] * g[j]);
            h = core::array::from_fn(|j| o[j] * c[j].tanh());
            *out = h;
        }
        output
    }

    /// Der Zellzustand `c` steht nicht in der Ausgabe und wird deshalb erst einmal durch eine
    /// Vorwärtsrechnung neu gewonnen. Dann geht es rückwärts: Mit `dh` (inklusive dem Gradienten vom
    /// nächsten Schritt) und `dc_next` (Gradient der Zelle vom nächsten Schritt), `tc = tanh(c)`:
    ///
    /// ```text
    /// do = dh ⊙ tc            dc = dc_next + dh ⊙ o ⊙ (1 − tc²)
    /// df = dc ⊙ c_alt        di = dc ⊙ g        dg = dc ⊙ i        dc_alt = dc ⊙ f
    /// d(ai) = di ⊙ i (1 − i)   d(af) = df ⊙ f (1 − f)   d(ag) = dg ⊙ (1 − g²)   d(ao) = do ⊙ o (1 − o)
    /// ```
    ///
    /// Die vier `d(a…)` gehen durch `gate_backward` auf die Gewichte, die Eingabe und `h_alt`.
    fn backward(
        &self,
        params: &Self::Params,
        grads: &mut Self::Params,
        input: &Self::Input,
        output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let ((input_gate, forget_gate), (candidate, output_gate)) = params;
        let ((grad_input_gate, grad_forget_gate), (grad_candidate, grad_output_gate)) = grads;

        // Zellzustände aller Schritte neu berechnen.
        let mut cells = [[T::ZERO; H]; STEPS];
        let mut c = [T::ZERO; H];
        for t in 0..STEPS {
            let h_prev = if t > 0 { output[t - 1] } else { [T::ZERO; H] };
            let (i, f, g, _) = Self::gates(params, &input[t], &h_prev);
            c = core::array::from_fn(|j| f[j] * c[j] + i[j] * g[j]);
            cells[t] = c;
        }

        let mut grad_input = [[T::ZERO; IN]; STEPS];
        let (mut dh_next, mut dc_next) = ([T::ZERO; H], [T::ZERO; H]);
        for t in (0..STEPS).rev() {
            let h_prev = if t > 0 { output[t - 1] } else { [T::ZERO; H] };
            let c_prev = if t > 0 { cells[t - 1] } else { [T::ZERO; H] };
            let x = &input[t];
            let (i, f, g, o) = Self::gates(params, x, &h_prev);
            let tanh_c: [T; H] = cells[t].map(Scalar::tanh);
            let dh: [T; H] = core::array::from_fn(|j| grad_output[t][j] + dh_next[j]);

            let d_o: [T; H] = core::array::from_fn(|j| dh[j] * tanh_c[j]);
            let dc: [T; H] = core::array::from_fn(|j| {
                dc_next[j] + dh[j] * o[j] * (T::ONE - tanh_c[j] * tanh_c[j])
            });

            let da_i: [T; H] = core::array::from_fn(|j| dc[j] * g[j] * i[j] * (T::ONE - i[j]));
            let da_f: [T; H] = core::array::from_fn(|j| dc[j] * c_prev[j] * f[j] * (T::ONE - f[j]));
            let da_g: [T; H] = core::array::from_fn(|j| dc[j] * i[j] * (T::ONE - g[j] * g[j]));
            let da_o: [T; H] = core::array::from_fn(|j| d_o[j] * o[j] * (T::ONE - o[j]));

            let (mut dx, mut dh_prev) = ([T::ZERO; IN], [T::ZERO; H]);
            gate_backward(
                input_gate,
                grad_input_gate,
                x,
                &h_prev,
                &da_i,
                &mut dx,
                &mut dh_prev,
            );
            gate_backward(
                forget_gate,
                grad_forget_gate,
                x,
                &h_prev,
                &da_f,
                &mut dx,
                &mut dh_prev,
            );
            gate_backward(
                candidate,
                grad_candidate,
                x,
                &h_prev,
                &da_g,
                &mut dx,
                &mut dh_prev,
            );
            gate_backward(
                output_gate,
                grad_output_gate,
                x,
                &h_prev,
                &da_o,
                &mut dx,
                &mut dh_prev,
            );

            grad_input[t] = dx;
            dh_next = dh_prev;
            dc_next = core::array::from_fn(|j| dc[j] * f[j]);
        }
        grad_input
    }

    /// `Wx` und `Wh` aller Tore aus dem `Initializer`, alle Biase 0 außer dem des Vergiss-Tors (1).
    fn init_params<I: Initializer>(&self, initializer: &I, rng: &mut Rng) -> Self::Params {
        let input_gate = init_gate(initializer, rng);
        let (weights, _) = init_gate::<IN, H, T, I>(initializer, rng);
        let forget_gate = (weights, [T::ONE; H]);
        let candidate = init_gate(initializer, rng);
        let output_gate = init_gate(initializer, rng);
        ((input_gate, forget_gate), (candidate, output_gate))
    }
}

/// Eine LSTM-Schicht.
#[cfg(any(feature = "std", feature = "libm"))]
pub type LstmLayer<const STEPS: usize, const IN: usize, const H: usize, T = f32> =
    Block<Lstm<STEPS, IN, H, T>>;
