use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::Values;

/// Ordnet dieselben Zahlen in eine andere Form um, ohne zu rechnen.
///
/// Typisch: ein Bild `[[[T; 28]; 28]; 1]` flachdrücken zu `[T; 784]`, damit eine Dense-Schicht
/// folgen kann, oder einen Vektor wieder zu einem Bild falten. Die Zahlen behalten ihre
/// Reihenfolge (siehe [`Values`]). Hat keine Parameter.
///
/// Beide Formen müssen gleich viele Zahlen haben. Das prüft der Compiler:
///
/// ```compile_fail
/// use neuronws_layers::reshape::Reshape;
///
/// // 2·3 = 6 Zahlen, aber 5 gewünscht.
/// let _ = Reshape::<[[f32; 3]; 2], [f32; 5]>::new();
/// ```
///
/// ```
/// use neuronws_layers::block::Block;
/// use neuronws_layers::reshape::Reshape;
/// use neuronws_layers::traits::ForwardPath;
///
/// let flatten = Block::from_parts(Reshape::<[[f32; 3]; 2], [f32; 6]>::new(), ());
/// assert_eq!(
///     flatten.forward(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
///     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
/// );
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Reshape<I: Values<T>, O: Values<T>, T: Scalar = f32> {
    shapes: PhantomData<(I, O, T)>,
}

impl<I: Values<T>, O: Values<T>, T: Scalar> Reshape<I, O, T> {
    /// # Compile-Fehler
    /// Wenn `I` und `O` nicht gleich viele Zahlen haben.
    pub const fn new() -> Self {
        const {
            assert!(
                I::LEN == O::LEN,
                "Reshape: beide Formen brauchen gleich viele Zahlen"
            )
        };
        Self {
            shapes: PhantomData,
        }
    }
}

impl<I: Values<T>, O: Values<T>, T: Scalar> Default for Reshape<I, O, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Values<T>, O: Values<T>, T: Scalar> Operation for Reshape<I, O, T> {
    type Scalar = T;
    type Input = I;
    type Output = O;
    type Params = ();

    fn forward(&self, _params: &(), input: &I) -> O {
        copy_values(input)
    }

    /// Der Gradient wandert unverändert zurück, nur wieder in der Form der Eingabe.
    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &I,
        _output: &O,
        grad_output: &O,
    ) -> I {
        copy_values(grad_output)
    }

    fn init_params<In: Initializer>(&self, _initializer: &In, _rng: &mut Rng) {}
}

fn copy_values<T: Scalar, From: Values<T>, To: Values<T>>(from: &From) -> To {
    let mut to = To::zeros();
    for i in 0..From::LEN {
        *to.value_mut(i) = from.value(i);
    }
    to
}

/// Eine Umformungs-Schicht von `I` nach `O`.
pub type ReshapeLayer<I, O, T = f32> = Block<Reshape<I, O, T>>;

impl<I: Values<T>, O: Values<T>, T: Scalar> Block<Reshape<I, O, T>> {
    pub fn new() -> Self {
        Self::from_parts(Reshape::new(), ())
    }
}

impl<I: Values<T>, O: Values<T>, T: Scalar> Default for Block<Reshape<I, O, T>> {
    fn default() -> Self {
        Self::new()
    }
}
