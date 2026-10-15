//! `derivative_with_output` liefert dasselbe wie `derivative`, spart aber die Neuberechnung.

use core::cell::Cell;

use neuronws_layers::activation::{Activation, Identity, Relu, Step};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::scalar::Scalar;
use neuronws_layers::traits::{BackwardPathMut, ForwardPathMut};

const ZS: [f32; 6] = [-3.0, -1.0, -0.2, 0.4, 1.5, 4.0];

fn assert_same<A: Activation>(activation: &A) {
    for z in ZS {
        let y = activation.apply(z);
        assert_eq!(
            activation.derivative_with_output(z, y),
            activation.derivative(z),
            "bei z = {z}"
        );
    }
}

#[test]
fn default_implementation_agrees_with_derivative() {
    assert_same(&Identity);
    assert_same(&Step);
    assert_same(&Relu);
}

#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn sigmoid_and_tanh_agree_with_derivative() {
    use neuronws_layers::activation::{Sigmoid, Tanh};
    for z in ZS {
        for (a, b) in [
            (
                Sigmoid.derivative_with_output(z, Sigmoid.apply(z)),
                Sigmoid.derivative(z),
            ),
            (
                Tanh.derivative_with_output(z, Tanh.apply(z)),
                Tanh.derivative(z),
            ),
        ] {
            assert!((a - b).abs() < 1e-6, "bei z = {z}: {a} gegen {b}");
        }
    }
}

/// Zählt, wie oft `apply` aufgerufen wird.
struct CountingActivation<'a> {
    applies: &'a Cell<usize>,
}

impl Activation for CountingActivation<'_> {
    fn apply<T: Scalar>(&self, z: T) -> T {
        self.applies.set(self.applies.get() + 1);
        z
    }

    /// Die naive Ableitung rechnet `apply` noch einmal aus, wie es Sigmoid früher tat.
    fn derivative<T: Scalar>(&self, z: T) -> T {
        self.apply(z);
        T::ONE
    }

    fn derivative_with_output<T: Scalar>(&self, _z: T, _y: T) -> T {
        T::ONE
    }
}

#[test]
fn activation_layer_backward_does_not_recompute_apply() {
    let applies = Cell::new(0);
    let mut layer = ActivationLayer::<4, _>::new(CountingActivation { applies: &applies });

    layer.forward_mut(&[1.0_f32, 2.0, 3.0, 4.0]);
    assert_eq!(applies.get(), 4);

    layer.backward_mut(&[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(
        applies.get(),
        4,
        "backward darf apply nicht erneut aufrufen"
    );
}
