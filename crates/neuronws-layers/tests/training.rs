//! Das Netz lernt wirklich: lineare Regression `y = 2x + 1`.

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

#[test]
fn sgd_step_moves_parameters_against_the_gradient_and_clears_it() {
    let mut layer = DenseLayer::new([[1.0]], [0.0]);
    layer.forward_mut(&[2.0]);
    layer.backward_mut(&[1.0]); // grad_w = 1 * 2, grad_b = 1

    Sgd::new(0.5).step(&mut layer, 1);

    assert_eq!(layer.weights(), &[[1.0 - 0.5 * 2.0]]);
    assert_eq!(layer.bias(), &[-0.5]);
    assert_eq!(layer.grad_weights(), &[[0.0]]);
    assert_eq!(layer.grad_bias(), &[0.0]);
}

#[test]
fn dense_layer_learns_a_line() {
    let mut layer = DenseLayer::new([[0.0]], [0.0]);
    let sgd = Sgd::new(0.05);
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x| ([x], [2.0 * x + 1.0]));

    for _ in 0..300 {
        for (x, target) in &data {
            let y = *layer.forward_mut(x);
            let gradient = MeanSquaredError.gradient(&y, target);
            layer.backward_mut(&gradient);
        }
        sgd.step(&mut layer, data.len());
    }

    let [[w]] = *layer.weights();
    let [b] = *layer.bias();
    assert!((w - 2.0).abs() < 1e-2, "w = {w}");
    assert!((b - 1.0).abs() < 1e-2, "b = {b}");
}

#[test]
fn network_learns_xor() {
    // Feste Startwerte, damit der Test jedes Mal gleich verläuft.
    let mut net = DenseLayer::new(
        [[0.5, -0.4], [-0.3, 0.6], [0.8, 0.2], [-0.6, -0.7]],
        [0.1, 0.1, -0.1, 0.2],
    )
    .then(ActivationLayer::<4, _>::new(Relu))
    .then(DenseLayer::new([[0.5, -0.5, 0.3, -0.2]], [0.0]));
    let sgd = Sgd::new(0.05);
    let data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    for _ in 0..5000 {
        for (x, target) in &data {
            let y = *net.forward_mut(x);
            net.backward_mut(&MeanSquaredError.gradient(&y, target));
        }
        sgd.step(&mut net, data.len());
    }

    for (x, [target]) in &data {
        let [y] = net.forward(x);
        assert!((y - target).abs() < 0.05, "{x:?}: {y} statt {target}");
    }
}
