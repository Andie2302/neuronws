//! Eine einzelne Dense-Schicht: 2 Eingänge, 3 Neuronen.
//!
//! Start: `cargo run -p neuronws-layers --example dense_forward`

use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::traits::ForwardPath;

fn main() {
    // Jede Zeile ist die Gewichtszeile eines Neurons.
    let layer =
        DenseLayer::<2, 3, f32>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [0.5, 0.0, -1.0]);

    let input = [1.0, 10.0];
    let output = layer.forward(&input);

    println!("Eingabe: {input:?}");
    println!("Ausgabe: {output:?}");
    // Neuron 0: 1*1 + 2*10 + 0.5 = 21.5
}
