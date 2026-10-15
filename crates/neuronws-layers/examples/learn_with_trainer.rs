//! Mini-Batches und gemischte Reihenfolge mit `train_epoch`: ein Netz lernt XOR.
//!
//! Start: `cargo run -p neuronws-layers --features libm --example learn_with_trainer`

use neuronws_layers::activation::Tanh;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::training::{mean_loss, train_epoch};

fn main() {
    let mut data: [([f32; 2], [f32; 1]); 4] = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    let mut rng = Rng::new(42);
    let mut net = DenseLayer::<2, 4>::xavier(&mut rng)
        .then(ActivationLayer::<4, _>::new(Tanh))
        .then(DenseLayer::<4, 1>::xavier(&mut rng));
    let mut sgd = Sgd::new(0.1);

    for epoch in 0..=2000 {
        rng.shuffle(&mut data);
        let loss = train_epoch(&mut net, &mut sgd, &MeanSquaredError, &data, 2);
        if epoch % 400 == 0 {
            println!("Epoche {epoch:4}: Verlust {loss:.6}");
        }
    }
    println!(
        "Endverlust: {:.6}",
        mean_loss(&net, &MeanSquaredError, &data)
    );
}
