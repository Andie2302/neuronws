use core::marker::PhantomData;

use crate::activation::Activation;
use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;

/// Wendet eine Aktivierungsfunktion elementweise auf `N` Werte an. Hat keine Parameter.
#[derive(Debug, Clone, Copy)]
pub struct Elementwise<const N: usize, A: Activation, T: Scalar = f32> {
    activation: A,
    scalar: PhantomData<T>,
}

impl<const N: usize, A: Activation, T: Scalar> Elementwise<N, A, T> {
    pub const fn new(activation: A) -> Self {
        Self {
            activation,
            scalar: PhantomData,
        }
    }
}

impl<const N: usize, A: Activation, T: Scalar> Operation for Elementwise<N, A, T> {
    type Scalar = T;
    type Input = [T; N];
    type Output = [T; N];
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        input.map(|z| self.activation.apply(z))
    }

    /// Kettenregel elementweise: `∂L/∂z[i] = g[i] · φ'(z[i])`.
    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        input: &Self::Input,
        output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        core::array::from_fn(|i| {
            grad_output[i] * self.activation.derivative_with_output(input[i], output[i])
        })
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Eine Aktivierungsschicht: wendet `φ` auf jeden der `N` Werte an.
pub type ActivationLayer<const N: usize, A, T = f32> = Block<Elementwise<N, A, T>>;

impl<const N: usize, A: Activation, T: Scalar> Block<Elementwise<N, A, T>> {
    pub fn new(activation: A) -> Self {
        Self::from_parts(Elementwise::new(activation), ())
    }
}
