//! Backward-Pass der Aktivierungsschicht gegen numerische Gradienten.

use neuronws_layers::activation::{Activation, Identity, Relu};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

// Alle Werte liegen weit weg von 0, dem Knick von ReLU.
const Z: [f32; 4] = [-1.5, 0.7, 2.0, -0.3];
const TARGET: [f32; 4] = [0.5, 0.5, 0.5, 0.5];
const EPS: f32 = 1e-2;
const TOLERANCE: f32 = 1e-3;

/// L = ½ Σ (φ(z) − t)²
fn loss<A: Activation>(layer: &ActivationLayer<4, A>, z: [f32; 4]) -> f32 {
    let y = layer.forward(&z);
    0.5 * y
        .iter()
        .zip(TARGET)
        .map(|(y, t)| (y - t) * (y - t))
        .sum::<f32>()
}

fn check_gradients<A: Activation>(make: impl Fn() -> A) {
    let mut layer = ActivationLayer::<4, A>::new(make());
    let y = *layer.forward_mut(&Z);
    let grad_output = core::array::from_fn(|i| y[i] - TARGET[i]);
    let grad_input = *layer.backward_mut(&grad_output);

    let reference = ActivationLayer::<4, A>::new(make());
    for (i, analytic) in grad_input.iter().enumerate() {
        let mut up = Z;
        up[i] += EPS;
        let mut down = Z;
        down[i] -= EPS;
        let numeric = (loss(&reference, up) - loss(&reference, down)) / (2.0 * EPS);
        assert!(
            (analytic - numeric).abs() < TOLERANCE,
            "i = {i}: analytisch {analytic}, numerisch {numeric}"
        );
    }
}

#[test]
fn relu_gradients_match_numeric_gradients() {
    check_gradients(|| Relu);
}

#[test]
fn identity_gradients_match_numeric_gradients() {
    check_gradients(|| Identity);
}

#[test]
fn forward_mut_matches_forward() {
    let mut layer = ActivationLayer::<4, _>::new(Relu);
    let expected = layer.forward(&Z);
    assert_eq!(layer.forward_mut(&Z), &expected);
}
