//! Eindimensionale Faltung für Folgen: [`Conv1d`].
//!
//! Ein Kern der Länge `K` wandert entlang einer Folge (Zeitreihe, Audio, Text als Zahlen). Die
//! Folge hat `C_IN` Kanäle (zum Beispiel mehrere Sensoren) und `L` Zeitschritte,
//! `input[kanal][zeitschritt]`. Schrittweite, Rand und Dilation funktionieren wie bei
//! [`Conv2d`](crate::conv::Conv2d), nur entlang einer Achse; die Formel für die Ausgabelänge ist
//! [`conv_output_size`].

use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::conv::{conv_output_size, tap};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::Values;

/// Die Kerne einer 1D-Faltung: `kernels[ausgabekanal][eingabekanal][position]`.
pub type Kernels1d<const C_OUT: usize, const C_IN: usize, const K: usize, T = f32> =
    [[[T; K]; C_IN]; C_OUT];

/// Eindimensionale Faltung:
///
/// ```text
/// y[o][t] = b[o] + Σ_{i,k} K[o][i][k] · x[i][t·STRIDE + k·DIL − PAD]
/// ```
///
/// Werte außerhalb der Folge (im Rand) zählen als 0. Die Ausgabelänge ist
/// `OL = (L + 2·PAD − DIL·(K − 1) − 1) / STRIDE + 1`, siehe [`conv_output_size`], und steht wie
/// bei der 2D-Faltung als eigene Zahl da. Für den einfachen Fall gibt es [`Conv1dValid`] und
/// das Makro [`conv1d!`](crate::conv1d).
///
/// ```
/// use neuronws_layers::conv1d::Conv1dValidLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// // Ein Kanal, Folge der Länge 5, Kern [1, -1]: die Differenz benachbarter Werte
/// let conv = Conv1dValidLayer::<1, 1, 5, 2, 4>::new([[[1.0, -1.0]]], [0.0]);
/// assert_eq!(conv.forward(&[[1.0, 4.0, 9.0, 16.0, 25.0]]), [[-3.0, -5.0, -7.0, -9.0]]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Conv1d<
    const C_IN: usize,
    const C_OUT: usize,
    const L: usize,
    const K: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OL: usize,
    T: Scalar = f32,
> {
    scalar: PhantomData<T>,
}

/// Die einfache 1D-Faltung ohne Rand, mit Schrittweite 1 und ohne Dilation: `OL = L − K + 1`.
pub type Conv1dValid<
    const C_IN: usize,
    const C_OUT: usize,
    const L: usize,
    const K: usize,
    const OL: usize,
    T = f32,
> = Conv1d<C_IN, C_OUT, L, K, 1, 0, 1, OL, T>;

impl<
    const C_IN: usize,
    const C_OUT: usize,
    const L: usize,
    const K: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OL: usize,
    T: Scalar,
> Conv1d<C_IN, C_OUT, L, K, STRIDE, PAD, DIL, OL, T>
{
    /// # Compile-Fehler
    /// Wenn Kern, Schrittweite oder Dilation 0 sind, der Kern (mit Dilation) länger als die Folge
    /// mit Rand ist oder `OL` nicht zur Formel passt.
    pub const fn new() -> Self {
        const {
            assert!(
                OL == conv_output_size(L, K, STRIDE, PAD, DIL),
                "Conv1d: OL muss (L + 2·PAD − DIL·(K − 1) − 1) / STRIDE + 1 sein, siehe \
                 conv_output_size"
            );
        };
        Self {
            scalar: PhantomData,
        }
    }
}

impl<
    const C_IN: usize,
    const C_OUT: usize,
    const L: usize,
    const K: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OL: usize,
    T: Scalar,
> Default for Conv1d<C_IN, C_OUT, L, K, STRIDE, PAD, DIL, OL, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    const C_IN: usize,
    const C_OUT: usize,
    const L: usize,
    const K: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OL: usize,
    T: Scalar,
> Operation for Conv1d<C_IN, C_OUT, L, K, STRIDE, PAD, DIL, OL, T>
{
    type Scalar = T;
    type Input = [[T; L]; C_IN];
    type Output = [[T; OL]; C_OUT];
    type Params = (Kernels1d<C_OUT, C_IN, K, T>, [T; C_OUT]);

    fn forward(&self, (kernels, bias): &Self::Params, input: &Self::Input) -> Self::Output {
        let mut output = [[T::ZERO; OL]; C_OUT];
        for (o, channel) in output.iter_mut().enumerate() {
            for (t, cell) in channel.iter_mut().enumerate() {
                let mut sum = bias[o];
                for (i, signal) in input.iter().enumerate() {
                    for (k, &weight) in kernels[o][i].iter().enumerate() {
                        if let Some(position) = tap(t, k, STRIDE, DIL, PAD, L) {
                            sum += weight * signal[position];
                        }
                    }
                }
                *cell = sum;
            }
        }
        output
    }

    /// Jede Ausgabestelle `(o, t)` mit Gradient `g` trägt über jeden Kernpunkt `(i, k)` bei, der
    /// in der Folge liegt (Position `s`):
    /// `∂L/∂b[o] += g`, `∂L/∂K[o][i][k] += g · x[i][s]`, `∂L/∂x[i][s] += K[o][i][k] · g`.
    fn backward(
        &self,
        (kernels, _): &Self::Params,
        (grad_kernels, grad_bias): &mut Self::Params,
        input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let mut grad_input = [[T::ZERO; L]; C_IN];
        for (o, grad_channel) in grad_output.iter().enumerate() {
            for (t, &g) in grad_channel.iter().enumerate() {
                grad_bias[o] += g;
                for (i, signal) in input.iter().enumerate() {
                    for k in 0..K {
                        if let Some(position) = tap(t, k, STRIDE, DIL, PAD, L) {
                            grad_kernels[o][i][k] += g * signal[position];
                            grad_input[i][position] += kernels[o][i][k] * g;
                        }
                    }
                }
            }
        }
        grad_input
    }

    /// Kerne aus dem `Initializer`, Bias 0. Jeder Ausgabewert sieht `C_IN·K` Eingaben (`fan_in`),
    /// jeder Eingabewert beeinflusst `C_OUT·K` Ausgaben (`fan_out`).
    fn init_params<I: Initializer>(&self, initializer: &I, rng: &mut Rng) -> Self::Params {
        let mut kernels = <Kernels1d<C_OUT, C_IN, K, T> as Values<T>>::zeros();
        initializer.fill(rng, &mut kernels, C_IN * K, C_OUT * K);
        (kernels, [T::ZERO; C_OUT])
    }
}

/// Eine 1D-Faltungsschicht.
pub type Conv1dLayer<
    const C_IN: usize,
    const C_OUT: usize,
    const L: usize,
    const K: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OL: usize,
    T = f32,
> = Block<Conv1d<C_IN, C_OUT, L, K, STRIDE, PAD, DIL, OL, T>>;

/// Eine einfache 1D-Faltungsschicht ohne Rand, mit Schrittweite 1 und ohne Dilation.
pub type Conv1dValidLayer<
    const C_IN: usize,
    const C_OUT: usize,
    const L: usize,
    const K: usize,
    const OL: usize,
    T = f32,
> = Block<Conv1dValid<C_IN, C_OUT, L, K, OL, T>>;

impl<
    const C_IN: usize,
    const C_OUT: usize,
    const L: usize,
    const K: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OL: usize,
    T: Scalar,
> Block<Conv1d<C_IN, C_OUT, L, K, STRIDE, PAD, DIL, OL, T>>
{
    pub fn new(kernels: Kernels1d<C_OUT, C_IN, K, T>, bias: [T; C_OUT]) -> Self {
        Self::from_parts(Conv1d::new(), (kernels, bias))
    }

    pub const fn kernels(&self) -> &Kernels1d<C_OUT, C_IN, K, T> {
        &self.params().0
    }

    pub const fn bias(&self) -> &[T; C_OUT] {
        &self.params().1
    }

    pub const fn grad_kernels(&self) -> &Kernels1d<C_OUT, C_IN, K, T> {
        &self.grads().0
    }

    pub const fn grad_bias(&self) -> &[T; C_OUT] {
        &self.grads().1
    }
}

/// Schreibt den Typ einer [`Conv1d`] mit allen Zahlen hin und rechnet `OL` selbst aus.
///
/// ```
/// use neuronws_layers::block::Block;
///
/// // 2 Kanäle rein, 8 raus, Länge 100, Kern 5, Schrittweite 2, Rand 2: ergibt Länge 50.
/// type Layer = Block<neuronws_layers::conv1d!(in: 2, out: 8, len: 100, kernel: 5, stride: 2, padding: 2)>;
/// let layer = Layer::new([[[0.0; 5]; 2]; 8], [0.0; 8]);
/// # let _ = layer;
/// ```
///
/// Die Angaben stehen in dieser Reihenfolge, `stride`, `padding` und `dilation` dürfen von hinten
/// weggelassen werden (Standard: Schrittweite 1, Rand 0, Dilation 1), wie bei
/// [`conv2d!`](crate::conv2d).
#[macro_export]
macro_rules! conv1d {
    (in: $c_in:expr, out: $c_out:expr, len: $len:expr, kernel: $k:expr) => {
        $crate::conv1d!(in: $c_in, out: $c_out, len: $len, kernel: $k, stride: 1, padding: 0, dilation: 1)
    };
    (in: $c_in:expr, out: $c_out:expr, len: $len:expr, kernel: $k:expr, stride: $stride:expr) => {
        $crate::conv1d!(in: $c_in, out: $c_out, len: $len, kernel: $k, stride: $stride, padding: 0, dilation: 1)
    };
    (in: $c_in:expr, out: $c_out:expr, len: $len:expr, kernel: $k:expr, stride: $stride:expr, padding: $pad:expr) => {
        $crate::conv1d!(in: $c_in, out: $c_out, len: $len, kernel: $k, stride: $stride, padding: $pad, dilation: 1)
    };
    (in: $c_in:expr, out: $c_out:expr, len: $len:expr, kernel: $k:expr, stride: $stride:expr, padding: $pad:expr, dilation: $dil:expr) => {
        $crate::conv1d::Conv1d<
            { $c_in }, { $c_out }, { $len }, { $k }, { $stride }, { $pad }, { $dil },
            { $crate::conv::conv_output_size($len, $k, $stride, $pad, $dil) },
        >
    };
}
