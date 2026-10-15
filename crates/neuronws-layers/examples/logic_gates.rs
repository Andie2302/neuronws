//! Logikgatter aus Dense + Step: jedes ist ein klassisches Perceptron.
//!
//! Start: `cargo run -p neuronws-layers --example logic_gates`

use neuronws_layers::activation::Step;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::Chain;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::traits::ForwardPath;

/// Ein Perceptron: eine Dense-Schicht mit einem Neuron, gefolgt von `Step`.
fn perceptron(
    weights: [f32; 2],
    bias: f32,
) -> impl ForwardPath<ForwardInput = [f32; 2], ForwardOutput = [f32; 1]> {
    Chain::new(
        DenseLayer::new([weights], [bias]),
        ActivationLayer::<1, _>::new(Step),
    )
}

fn print_table(
    name: &str,
    gate: &impl ForwardPath<ForwardInput = [f32; 2], ForwardOutput = [f32; 1]>,
) {
    println!("{name}");
    for a in [0.0, 1.0] {
        for b in [0.0, 1.0] {
            let [y] = gate.forward(&[a, b]);
            println!("  {a} {b} -> {y}");
        }
    }
}

fn main() {
    print_table("AND", &perceptron([1.0, 1.0], -1.5));
    print_table("OR", &perceptron([1.0, 1.0], -0.5));
    print_table("NAND", &perceptron([-1.0, -1.0], 1.5));
}
