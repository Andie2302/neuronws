use core::marker::PhantomData;

use crate::activation::Activation;
use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::Values;

/// Wendet eine Aktivierungsfunktion elementweise auf jeden Wert einer Gruppe `V` an.
/// Hat keine Parameter.
///
/// `V` darf jede Form haben: ein Vektor `[T; N]`, eine Matrix, ein Bild mit Kanälen.
#[derive(Debug, Clone, Copy)]
pub struct Elementwise<V: Values<T>, A: Activation, T: Scalar = f32> {
    activation: A,
    shape: PhantomData<(V, T)>,
}

impl<V: Values<T>, A: Activation, T: Scalar> Elementwise<V, A, T> {
    pub const fn new(activation: A) -> Self {
        Self {
            activation,
            shape: PhantomData,
        }
    }
}

impl<V: Values<T>, A: Activation, T: Scalar> Operation for Elementwise<V, A, T> {
    type Scalar = T;
    type Input = V;
    type Output = V;
    type Params = ();

    fn forward(&self, _params: &(), input: &V) -> V {
        let mut output = V::zeros();
        for i in 0..V::LEN {
            *output.value_mut(i) = self.activation.apply(input.value(i));
        }
        output
    }

    /// Kettenregel elementweise: `∂L/∂z[i] = g[i] · φ'(z[i])`.
    fn backward(&self, _params: &(), _grads: &mut (), input: &V, output: &V, grad_output: &V) -> V {
        let mut grad_input = V::zeros();
        for i in 0..V::LEN {
            *grad_input.value_mut(i) = grad_output.value(i)
                * self
                    .activation
                    .derivative_with_output(input.value(i), output.value(i));
        }
        grad_input
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Eine Aktivierungsschicht: wendet `φ` auf jeden der `N` Werte eines Vektors an.
///
/// Für andere Formen `Block<Elementwise<V, A, T>>` benutzen.
pub type ActivationLayer<const N: usize, A, T = f32> = Block<Elementwise<[T; N], A, T>>;

impl<V: Values<T>, A: Activation, T: Scalar> Block<Elementwise<V, A, T>> {
    pub fn new(activation: A) -> Self {
        Self::from_parts(Elementwise::new(activation), ())
    }
}
