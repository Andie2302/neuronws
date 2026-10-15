//! Backward-Pass durch eine ganze Kette: Dense → ReLU → Dense.

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

const W1: [[f32; 2]; 3] = [[0.5, -1.0], [2.0, 0.3], [-0.7, 1.1]];
const B1: [f32; 3] = [0.1, -0.2, 0.3];
const W2: [[f32; 3]; 1] = [[1.0, -2.0, 0.5]];
const B2: [f32; 1] = [0.2];
// Die Werte vor ReLU sind [1.35, 2.65, -1.3], also weit weg vom Knick bei 0.
const X: [f32; 2] = [1.5, -0.5];
const TARGET: f32 = 1.0;
const EPS: f32 = 1e-2;
const TOLERANCE: f32 = 1e-3;

macro_rules! net {
    () => {
        DenseLayer::new(W1, B1)
            .then(ActivationLayer::<3, _>::new(Relu))
            .then(DenseLayer::new(W2, B2))
    };
}

/// L = ½ (y − t)²
fn loss(x: [f32; 2]) -> f32 {
    let [y] = net!().forward(&x);
    0.5 * (y - TARGET) * (y - TARGET)
}

#[test]
fn forward_mut_matches_forward() {
    let mut chain = net!();
    let expected = chain.forward(&X);
    assert_eq!(chain.forward_mut(&X), &expected);
}

#[test]
fn input_gradient_through_the_chain_matches_numeric_gradient() {
    let mut chain = net!();
    let [y] = *chain.forward_mut(&X);
    let grad_input = *chain.backward_mut(&[y - TARGET]);

    for (i, analytic) in grad_input.iter().enumerate() {
        let mut up = X;
        up[i] += EPS;
        let mut down = X;
        down[i] -= EPS;
        let numeric = (loss(up) - loss(down)) / (2.0 * EPS);
        assert!(
            (analytic - numeric).abs() < TOLERANCE,
            "i = {i}: analytisch {analytic}, numerisch {numeric}"
        );
    }
}
