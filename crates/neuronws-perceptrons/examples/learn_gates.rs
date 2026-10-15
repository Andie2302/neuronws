//! Ein Perceptron lernt Logikgatter aus Beispielen, und scheitert an XOR.
//!
//! Start: `cargo run -p neuronws-perceptrons --example learn_gates`

use neuronws_perceptrons::Perceptron;

type Data = [([f32; 2], f32); 4];

const AND: Data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 0.0),
    ([1.0, 0.0], 0.0),
    ([1.0, 1.0], 1.0),
];
const OR: Data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 1.0),
];
const XOR: Data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

fn main() {
    for (name, data) in [("AND", &AND), ("OR", &OR), ("XOR", &XOR)] {
        let mut perceptron = Perceptron::<2>::new([0.0, 0.0], 0.0);
        let result = perceptron.fit(data, 1000, 0.1);

        match result {
            Some(epochs) => println!(
                "{name}: gelernt nach {epochs} Runden, w = {:?}, b = {}",
                perceptron.weights(),
                perceptron.bias()
            ),
            None => println!("{name}: nach 1000 Runden immer noch Fehler (nicht linear trennbar)"),
        }
    }
}
