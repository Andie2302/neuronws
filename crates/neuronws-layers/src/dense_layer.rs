use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::init::Xavier;
use crate::random::Rng;
use crate::scalar::Scalar;

/// Die lineare Abbildung `y = W·x + b` von `IN` Eingängen auf `OUT` Neuronen.
///
/// Die Parameter sind die Gewichte (`OUT` Zeilen zu je `IN` Werten) und der Bias (`OUT` Werte).
/// `weights[j]` ist die Gewichtszeile des Neurons `j`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Linear<const IN: usize, const OUT: usize, T: Scalar = f32>(PhantomData<T>);

impl<const IN: usize, const OUT: usize, T: Scalar> Linear<IN, OUT, T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const IN: usize, const OUT: usize, T: Scalar> Operation for Linear<IN, OUT, T> {
    type Scalar = T;
    type Input = [T; IN];
    type Output = [T; OUT];
    type Params = ([[T; IN]; OUT], [T; OUT]);

    /// `y[j] = Σ_i W[j][i] * x[i] + b[j]`
    fn forward(&self, (weights, bias): &Self::Params, input: &Self::Input) -> Self::Output {
        core::array::from_fn(|j| {
            let dot = weights[j]
                .iter()
                .zip(input)
                .fold(T::ZERO, |sum, (&w, &x)| sum + w * x);
            dot + bias[j]
        })
    }

    /// `∂L/∂W[j][i] = g[j] · x[i]`, `∂L/∂b[j] = g[j]`, `∂L/∂x[i] = Σ_j W[j][i] · g[j]`
    fn backward(
        &self,
        (weights, _): &Self::Params,
        (grad_weights, grad_bias): &mut Self::Params,
        input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        for (j, &g) in grad_output.iter().enumerate() {
            grad_bias[j] += g;
            for (grad_w, &x) in grad_weights[j].iter_mut().zip(input) {
                *grad_w += g * x;
            }
        }

        core::array::from_fn(|i| {
            weights
                .iter()
                .zip(grad_output)
                .fold(T::ZERO, |sum, (row, &g)| sum + row[i] * g)
        })
    }

    /// Gewichte aus dem `Initializer`, Bias 0.
    fn init_params<I: Initializer>(&self, initializer: &I, rng: &mut Rng) -> Self::Params {
        let mut weights = [[T::ZERO; IN]; OUT];
        initializer.fill(rng, &mut weights, IN, OUT);
        (weights, [T::ZERO; OUT])
    }
}

/// Voll verbundene Schicht: `OUT` Neuronen, die alle `IN` Eingänge sehen.
pub type DenseLayer<const IN: usize, const OUT: usize, T = f32> = Block<Linear<IN, OUT, T>>;

impl<const IN: usize, const OUT: usize, T: Scalar> Block<Linear<IN, OUT, T>> {
    pub fn new(weights: [[T; IN]; OUT], bias: [T; OUT]) -> Self {
        Self::from_parts(Linear::new(), (weights, bias))
    }

    /// Gewichte gleichverteilt in `[-limit, limit]`, Bias 0.
    ///
    /// Zufällige Startwerte sind nötig, damit die Neuronen einer Schicht nicht alle dasselbe lernen.
    pub fn random(rng: &mut Rng, limit: T) -> Self {
        let weights = core::array::from_fn(|_| {
            core::array::from_fn(|_| limit * (T::TWO * T::from_f32(rng.next_f32()) - T::ONE))
        });
        Self::new(weights, [T::ZERO; OUT])
    }

    /// Wie `random`, mit der Xavier-Grenze `√(6 / (IN + OUT))`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn xavier(rng: &mut Rng) -> Self {
        Self::initialized(Linear::new(), &Xavier, rng)
    }

    pub const fn weights(&self) -> &[[T; IN]; OUT] {
        &self.params().0
    }

    pub const fn bias(&self) -> &[T; OUT] {
        &self.params().1
    }

    pub const fn grad_weights(&self) -> &[[T; IN]; OUT] {
        &self.grads().0
    }

    pub const fn grad_bias(&self) -> &[T; OUT] {
        &self.grads().1
    }
}
