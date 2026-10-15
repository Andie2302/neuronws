use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::Values;

/// Softmax über die `N` Einträge von `input`, einzeln für jede der `V::LEN` Stellen innerhalb
/// eines Eintrags. Gemeinsame Rechnung von [`Softmax`] (dort ist jeder Eintrag eine Zahl) und
/// [`SoftmaxAcross`].
///
/// Vor dem Potenzieren wird der größte Wert abgezogen. Das ändert das Ergebnis nicht, verhindert
/// aber, dass `exp` bei großen Eingaben überläuft.
fn softmax_across<T: Scalar, V: Values<T>, const N: usize>(input: &[V; N]) -> [V; N] {
    let mut output = [V::zeros(); N];
    let Some(first) = input.first() else {
        return output;
    };
    for j in 0..V::LEN {
        let max = input
            .iter()
            .fold(first.value(j), |max, entry| max.max(entry.value(j)));
        let mut sum = T::ZERO;
        for (out, entry) in output.iter_mut().zip(input) {
            let e = (entry.value(j) - max).exp();
            *out.value_mut(j) = e;
            sum += e;
        }
        for out in &mut output {
            *out.value_mut(j) = out.value(j) / sum;
        }
    }
    output
}

/// Rückwärts zu [`softmax_across`]: `∂L/∂z[i] = y[i] · (g[i] − Σ_k g[k] · y[k])`, einzeln für
/// jede Stelle `j`.
///
/// Das ist die Jacobi-Matrix von Softmax, angewandt auf `g`, ohne dass sie je als Matrix
/// aufgebaut wird.
fn softmax_across_backward<T: Scalar, V: Values<T>, const N: usize>(
    output: &[V; N],
    grad_output: &[V; N],
) -> [V; N] {
    let mut grad_input = [V::zeros(); N];
    for j in 0..V::LEN {
        let weighted = output
            .iter()
            .zip(grad_output)
            .fold(T::ZERO, |sum, (y, g)| sum + g.value(j) * y.value(j));
        for ((grad, y), g) in grad_input.iter_mut().zip(output).zip(grad_output) {
            *grad.value_mut(j) = y.value(j) * (g.value(j) - weighted);
        }
    }
    grad_input
}

/// Macht aus `N` beliebigen Werten eine Wahrscheinlichkeitsverteilung: `y[i] = e^z[i] / Σ_j e^z[j]`.
///
/// Alle Ausgaben liegen in (0, 1) und summieren sich zu 1. Anders als eine elementweise
/// Aktivierung hängt jede Ausgabe von allen Eingaben ab, deshalb ist Softmax eine eigene Operation.
/// Braucht `exp`, also das Feature `std` oder `libm`.
///
/// Für andere Formen gibt es [`SoftmaxAcross`] (Softmax über die äußerste Achse, zum Beispiel über
/// die Kanäle eines Bildes) und `Each` (Softmax pro Zeile einer Matrix).
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmax<const N: usize, T: Scalar = f32>(PhantomData<T>);

impl<const N: usize, T: Scalar> Softmax<N, T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const N: usize, T: Scalar> Operation for Softmax<N, T> {
    type Scalar = T;
    type Input = [T; N];
    type Output = [T; N];
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        softmax_across::<T, T, N>(input)
    }

    /// `∂L/∂z[i] = y[i] · (g[i] − Σ_j g[j] · y[j])`
    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &Self::Input,
        output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        softmax_across_backward::<T, T, N>(output, grad_output)
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Softmax als Schicht.
pub type SoftmaxLayer<const N: usize, T = f32> = Block<Softmax<N, T>>;

impl<const N: usize, T: Scalar> Block<Softmax<N, T>> {
    pub fn new() -> Self {
        Self::from_parts(Softmax::new(), ())
    }
}

impl<const N: usize, T: Scalar> Default for Block<Softmax<N, T>> {
    fn default() -> Self {
        Self::new()
    }
}

/// Softmax über die **äußerste Achse** einer beliebigen Form `[V; N]`: An jeder Stelle innerhalb
/// von `V` wird über die `N` Einträge softmaxt.
///
/// Die Bilder dieser Bibliothek sind `[Kanal; C]` mit einer Ebene `[[T; W]; H]` je Kanal. Mit
/// `V = [[T; W]; H]` wird an **jedem Pixel** über die `C` Kanäle ein Softmax gebildet: Das ist die
/// Ausgabe eines Netzes, das jedem Pixel eine Klasse zuordnet (Segmentierung). Für diesen Fall
/// gibt es die Abkürzung [`SoftmaxChannels`].
///
/// ```
/// use neuronws_layers::softmax_layer::SoftmaxChannelsLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// // 3 Klassen, ein 1×2-Bild: Pixel 0 sieht nach Klasse 2 aus, Pixel 1 gleichmäßig.
/// let layer = SoftmaxChannelsLayer::<3, 1, 2>::new();
/// let image = [[[0.0, 0.0]], [[0.0, 0.0]], [[5.0, 0.0]]];
/// let probabilities = layer.forward(&image);
/// assert!(probabilities[2][0][0] > 0.98);
/// assert!((probabilities[0][0][1] - 1.0 / 3.0).abs() < 1e-6);
/// ```
///
/// Braucht `exp`, also das Feature `std` oder `libm`.
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxAcross<V: Values<T>, const N: usize, T: Scalar = f32> {
    shape: PhantomData<(V, T)>,
}

impl<V: Values<T>, const N: usize, T: Scalar> SoftmaxAcross<V, N, T> {
    /// # Compile-Fehler
    /// Wenn `N` 0 ist (es gäbe nichts, worüber man normieren könnte).
    pub const fn new() -> Self {
        const { assert!(N >= 1, "SoftmaxAcross: es braucht mindestens einen Eintrag") };
        Self { shape: PhantomData }
    }
}

impl<V: Values<T>, const N: usize, T: Scalar> Default for SoftmaxAcross<V, N, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Values<T>, const N: usize, T: Scalar> Operation for SoftmaxAcross<V, N, T> {
    type Scalar = T;
    type Input = [V; N];
    type Output = [V; N];
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        softmax_across(input)
    }

    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &Self::Input,
        output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        softmax_across_backward(output, grad_output)
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Softmax über die `C` Kanäle eines Bildes mit `H` Zeilen und `W` Spalten, an jedem Pixel einzeln.
pub type SoftmaxChannels<const C: usize, const H: usize, const W: usize, T = f32> =
    SoftmaxAcross<[[T; W]; H], C, T>;

/// Die Schicht zu [`SoftmaxChannels`].
pub type SoftmaxChannelsLayer<const C: usize, const H: usize, const W: usize, T = f32> =
    Block<SoftmaxAcross<[[T; W]; H], C, T>>;

impl<V: Values<T>, const N: usize, T: Scalar> Block<SoftmaxAcross<V, N, T>> {
    pub fn new() -> Self {
        Self::from_parts(SoftmaxAcross::new(), ())
    }
}

impl<V: Values<T>, const N: usize, T: Scalar> Default for Block<SoftmaxAcross<V, N, T>> {
    fn default() -> Self {
        Self::new()
    }
}
