//! XOR kann ein einzelnes Perceptron nicht, zwei Schichten hintereinander schon.
//!
//! XOR(a, b) = AND(OR(a, b), NAND(a, b))
//!
//! Start: `cargo run -p neuronws-layers --example xor_network`

use neuronws_layers::activation::Step;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::Chain;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::traits::ForwardPath;

fn main() {
    // Verborgene Schicht: Neuron 0 = OR, Neuron 1 = NAND
    let hidden = Chain::new(
        DenseLayer::new([[1.0, 1.0], [-1.0, -1.0]], [-0.5, 1.5]),
        ActivationLayer::<2, _>::new(Step),
    );
    // Ausgabeschicht: AND der beiden verborgenen Werte
    let output = Chain::new(
        DenseLayer::new([[1.0, 1.0]], [-1.5]),
        ActivationLayer::<1, _>::new(Step),
    );
    let xor = Chain::new(hidden, output);

    println!("XOR");
    for a in [0.0, 1.0] {
        for b in [0.0, 1.0] {
            let [y] = xor.forward(&[a, b]);
            println!("  {a} {b} -> {y}");
        }
    }
}
