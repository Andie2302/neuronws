use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;

/// Skaliert und verschiebt jede Eingabe einzeln: `y[i] = w[i] · x[i] + b[i]`.
///
/// Das ist eine Dense-Schicht, bei der nur die Diagonale der Gewichtsmatrix übrig ist: Jeder
/// Eingang hat genau ein eigenes Gewicht und einen eigenen Bias, die Eingänge sehen einander
/// nicht. Sie hat `2·N` statt `N·N + N` Parameter. Nützlich, um nach einer Normalisierung wieder
/// eine lernbare Skalierung zu haben, oder als kleinster denkbarer Lernbaustein.
///
/// Am Anfang tut die Schicht nichts (`w = 1`, `b = 0`), egal welchen `Initializer` man übergibt:
/// Ein zufälliger Start würde die Eingaben sofort durcheinanderbringen, ohne dass es etwas
/// gebracht hätte.
///
/// ```
/// use neuronws_layers::diagonal::DiagonalLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// let layer = DiagonalLayer::new([2.0, -1.0, 0.5], [0.0, 1.0, 0.0]);
/// assert_eq!(layer.forward(&[3.0, 4.0, 8.0]), [6.0, -3.0, 4.0]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Diagonal<const N: usize, T: Scalar = f32>(PhantomData<T>);

impl<const N: usize, T: Scalar> Diagonal<N, T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const N: usize, T: Scalar> Operation for Diagonal<N, T> {
    type Scalar = T;
    type Input = [T; N];
    type Output = [T; N];
    /// `(w, b)`
    type Params = ([T; N], [T; N]);

    fn forward(&self, (weights, bias): &Self::Params, input: &Self::Input) -> Self::Output {
        core::array::from_fn(|i| weights[i] * input[i] + bias[i])
    }

    /// `∂L/∂w[i] = g[i] · x[i]`, `∂L/∂b[i] = g[i]`, `∂L/∂x[i] = g[i] · w[i]`
    fn backward(
        &self,
        (weights, _): &Self::Params,
        (grad_weights, grad_bias): &mut Self::Params,
        input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        for i in 0..N {
            grad_weights[i] += grad_output[i] * input[i];
            grad_bias[i] += grad_output[i];
        }
        core::array::from_fn(|i| grad_output[i] * weights[i])
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) -> Self::Params {
        ([T::ONE; N], [T::ZERO; N])
    }
}

/// Eine Diagonal-Schicht: ein Gewicht und ein Bias pro Eingang.
pub type DiagonalLayer<const N: usize, T = f32> = Block<Diagonal<N, T>>;

impl<const N: usize, T: Scalar> Block<Diagonal<N, T>> {
    pub fn new(weights: [T; N], bias: [T; N]) -> Self {
        Self::from_parts(Diagonal::new(), (weights, bias))
    }

    /// `w = 1`, `b = 0`: Die Schicht ändert zunächst nichts.
    pub fn identity() -> Self {
        Self::new([T::ONE; N], [T::ZERO; N])
    }

    pub const fn weights(&self) -> &[T; N] {
        &self.params().0
    }

    pub const fn bias(&self) -> &[T; N] {
        &self.params().1
    }

    pub const fn grad_weights(&self) -> &[T; N] {
        &self.grads().0
    }

    pub const fn grad_bias(&self) -> &[T; N] {
        &self.grads().1
    }
}
