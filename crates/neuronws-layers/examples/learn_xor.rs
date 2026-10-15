//! Ein Netz lernt XOR, ohne dass wir Gewichte von Hand eintragen.
//!
//! Zwei Dense-Schichten mit ReLU dazwischen. `Step` geht hier nicht, seine Ableitung ist überall 0.
//!
//! Start: `cargo run -p neuronws-layers --example learn_xor`

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::traits::{BackwardPathMut, ForwardPathMut};

fn main() {
    // Feste Startwerte, damit jeder Lauf gleich verläuft. Lauter gleiche Gewichte wären ein Fehler:
    // Dann würden alle verborgenen Neuronen dasselbe lernen.
    let mut net = DenseLayer::new(
        [[0.5, -0.4], [-0.3, 0.6], [0.8, 0.2], [-0.6, -0.7]],
        [0.1, 0.1, -0.1, 0.2],
    )
    .then(ActivationLayer::<4, _>::new(Relu))
    .then(DenseLayer::new([[0.5, -0.5, 0.3, -0.2]], [0.0]));

    let mut sgd = Sgd::new(0.05);
    let data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    for epoch in 0..=5000 {
        let mut total_loss = 0.0;
        for (x, target) in &data {
            let y = *net.forward_mut(x);
            total_loss += MeanSquaredError.value(&y, target);
            net.backward_mut(&MeanSquaredError.gradient(&y, target));
        }
        sgd.step(&mut net, data.len());

        if epoch % 1000 == 0 {
            println!(
                "Epoche {epoch:4}: Verlust {:.6}",
                total_loss / data.len() as f32
            );
        }
    }

    println!("XOR");
    for (x, target) in &data {
        let [y] = *net.forward_mut(x);
        println!("  {x:?} -> {y:.3}  (Ziel {})", target[0]);
    }
}
