//! Eine Dense-Schicht lernt aus Beispielen die Gerade `y = 2x + 1`.
//!
//! Start: `cargo run -p neuronws-layers --example learn_linear`

use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::traits::{BackwardPathMut, ForwardPathMut};

fn main() {
    // Startwerte: Gewicht 0, Bias 0. Das Netz weiß noch nichts.
    let mut layer = DenseLayer::new([[0.0]], [0.0]);
    let sgd = Sgd::new(0.05);
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x| ([x], [2.0 * x + 1.0]));

    for epoch in 0..=300 {
        let mut total_loss = 0.0;
        for (x, target) in &data {
            let y = *layer.forward_mut(x);
            total_loss += MeanSquaredError.value(&y, target);
            // Backward-Pass: die Gradienten sammeln sich in der Schicht
            layer.backward_mut(&MeanSquaredError.gradient(&y, target));
        }
        // Lernschritt: Parameter gegen den gemittelten Gradienten verschieben
        sgd.step(&mut layer, data.len());

        if epoch % 50 == 0 {
            let [[w]] = *layer.weights();
            let [b] = *layer.bias();
            println!(
                "Epoche {epoch:3}: Verlust {:.6}  w = {w:.4}  b = {b:.4}",
                total_loss / data.len() as f32
            );
        }
    }
    println!("Ziel:        w = 2.0000  b = 1.0000");
}
