//! XOR lernen aus zufälligen Startwerten. Braucht `exp`/`tanh`, also `std` oder `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::Tanh;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

#[test]
fn tanh_network_learns_xor_from_many_random_starts() {
    let data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    for seed in 0..10 {
        let mut rng = Rng::new(seed);
        let mut net = DenseLayer::<2, 4>::xavier(&mut rng)
            .then(ActivationLayer::<4, _>::new(Tanh))
            .then(DenseLayer::<4, 1>::xavier(&mut rng));
        let sgd = Sgd::new(0.1);

        for _ in 0..5000 {
            for (x, target) in &data {
                let y = *net.forward_mut(x);
                net.backward_mut(&MeanSquaredError.gradient(&y, target));
            }
            sgd.step(&mut net, data.len());
        }

        for (x, [target]) in &data {
            let [y] = net.forward(x);
            assert!(
                (y - target).abs() < 0.3,
                "Seed {seed}, {x:?}: {y} statt {target}"
            );
        }
    }
}
