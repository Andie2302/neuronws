//! Ein kleines Netz aus Dense → ReLU → Dense, mit der Chain verkettet.
//!
//! Start: `cargo run -p neuronws-layers --example relu_network`

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::Chain;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::traits::ForwardPath;

fn main() {
    // 2 Eingänge -> 3 verborgene Werte -> ReLU -> 1 Ausgang
    let net = Chain::new(
        Chain::new(
            DenseLayer::new([[1.0, -1.0], [-1.0, 1.0], [0.5, 0.5]], [0.0, 0.0, 0.0]),
            ActivationLayer::<3, _>::new(Relu),
        ),
        DenseLayer::new([[1.0, 1.0, 1.0]], [0.0]),
    );

    for input in [[2.0, 1.0], [1.0, 2.0], [-3.0, -3.0]] {
        println!("{input:?} -> {:?}", net.forward(&input));
    }
}
