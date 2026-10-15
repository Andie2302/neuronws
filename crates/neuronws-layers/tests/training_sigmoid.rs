//! XOR lernen mit Sigmoid. Braucht das Feature `std` oder `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::Sigmoid;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

#[test]
fn sigmoid_network_learns_xor() {
    let mut net = DenseLayer::new(
        [[0.5, -0.4], [-0.3, 0.6], [0.8, 0.2], [-0.6, -0.7]],
        [0.1, 0.1, -0.1, 0.2],
    )
    .then(ActivationLayer::<4, _>::new(Sigmoid))
    .then(DenseLayer::new([[0.5, -0.5, 0.3, -0.2]], [0.0]))
    .then(ActivationLayer::<1, _>::new(Sigmoid));
    let mut sgd = Sgd::new(1.0);
    let data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    for _ in 0..10_000 {
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
