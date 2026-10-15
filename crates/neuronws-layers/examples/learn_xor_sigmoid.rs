//! Das klassische XOR-Netz mit Sigmoid statt ReLU.
//!
//! Sigmoid braucht `exp`, deshalb das Feature `libm` (oder `std`).
//!
//! Start: `cargo run -p neuronws-layers --features libm --example learn_xor_sigmoid`

use neuronws_layers::activation::Sigmoid;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::traits::{BackwardPathMut, ForwardPathMut};

fn main() {
    let mut net = DenseLayer::new(
        [[0.5, -0.4], [-0.3, 0.6], [0.8, 0.2], [-0.6, -0.7]],
        [0.1, 0.1, -0.1, 0.2],
    )
    .then(ActivationLayer::<4, _>::new(Sigmoid))
    .then(DenseLayer::new([[0.5, -0.5, 0.3, -0.2]], [0.0]))
    .then(ActivationLayer::<1, _>::new(Sigmoid));

    let sgd = Sgd::new(1.0);
    let data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    for epoch in 0..=10_000 {
        let mut total_loss = 0.0;
        for (x, target) in &data {
            let y = *net.forward_mut(x);
            total_loss += MeanSquaredError.value(&y, target);
            net.backward_mut(&MeanSquaredError.gradient(&y, target));
        }
        sgd.step(&mut net, data.len());

        if epoch % 2000 == 0 {
            println!(
                "Epoche {epoch:5}: Verlust {:.6}",
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
