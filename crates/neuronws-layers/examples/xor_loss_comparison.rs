//! XOR mit Sigmoid: wie schnell lernt das Netz mit mittlerem quadratischem Fehler (MSE)
//! gegenüber der Kreuzentropie (BCE)?
//!
//! Start: `cargo run -p neuronws-layers --features libm --example xor_loss_comparison`

use neuronws_layers::activation::Sigmoid;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{BinaryCrossEntropy, Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

const DATA: [([f32; 2], [f32; 1]); 4] = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
];

/// Anzahl der Runden, bis alle vier Ausgaben auf der richtigen Seite von 0,5 und mindestens
/// 0,1 von der Schwelle entfernt liegen.
fn epochs_until_correct(loss: &impl Loss, learning_rate: f32) -> Option<usize> {
    let mut net = DenseLayer::new(
        [[0.5, -0.4], [-0.3, 0.6], [0.8, 0.2], [-0.6, -0.7]],
        [0.1, 0.1, -0.1, 0.2],
    )
    .then(ActivationLayer::<4, _>::new(Sigmoid))
    .then(DenseLayer::new([[0.5, -0.5, 0.3, -0.2]], [0.0]))
    .then(ActivationLayer::<1, _>::new(Sigmoid));
    let mut sgd = Sgd::new(learning_rate);

    for epoch in 0..20_000 {
        let all_right = DATA
            .iter()
            .all(|(x, [t])| (net.forward(x)[0] - t).abs() < 0.4);
        if all_right {
            return Some(epoch);
        }
        for (x, target) in &DATA {
            let y = *net.forward_mut(x);
            net.backward_mut(&loss.gradient(&y, target));
        }
        sgd.step(&mut net, DATA.len());
    }
    None
}

fn main() {
    for learning_rate in [0.3, 1.0] {
        println!("Lernrate {learning_rate}:");
        for (name, epochs) in [
            (
                "MSE",
                epochs_until_correct(&MeanSquaredError, learning_rate),
            ),
            (
                "BCE",
                epochs_until_correct(&BinaryCrossEntropy, learning_rate),
            ),
        ] {
            match epochs {
                Some(n) => println!("  {name}: richtig nach {n} Runden"),
                None => println!("  {name}: nach 20000 Runden noch nicht"),
            }
        }
    }
}
