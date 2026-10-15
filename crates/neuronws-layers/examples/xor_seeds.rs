//! XOR lernen aus zufälligen Startwerten: Tanh klappt immer, ReLU nicht.
//!
//! Bei ReLU können verborgene Neuronen "sterben": Ist ihre Eingabe für alle Beispiele negativ,
//! ist ihre Ableitung 0 und sie lernen nie wieder etwas.
//!
//! Start: `cargo run -p neuronws-layers --features libm --example xor_seeds`

use neuronws_layers::activation::{Activation, Relu, Tanh};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

const DATA: [([f32; 2], [f32; 1]); 4] = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
];

fn learns_xor(activation: impl Activation, seed: u64) -> bool {
    let mut rng = Rng::new(seed);
    let mut net = DenseLayer::<2, 4>::xavier(&mut rng)
        .then(ActivationLayer::<4, _>::new(activation))
        .then(DenseLayer::<4, 1>::xavier(&mut rng));
    let sgd = Sgd::new(0.1);

    for _ in 0..5000 {
        for (x, target) in &DATA {
            let y = *net.forward_mut(x);
            net.backward_mut(&MeanSquaredError.gradient(&y, target));
        }
        sgd.step(&mut net, DATA.len());
    }

    DATA.iter()
        .all(|(x, [target])| (net.forward(x)[0] - target).abs() < 0.3)
}

fn main() {
    println!("Startwert  ReLU  Tanh");
    let (mut relu_ok, mut tanh_ok) = (0, 0);
    for seed in 0..20 {
        let relu = learns_xor(Relu, seed);
        let tanh = learns_xor(Tanh, seed);
        relu_ok += usize::from(relu);
        tanh_ok += usize::from(tanh);
        println!("{seed:9}  {:4}  {:4}", mark(relu), mark(tanh));
    }
    println!("Erfolgreich: ReLU {relu_ok}/20, Tanh {tanh_ok}/20");
}

fn mark(ok: bool) -> &'static str {
    if ok { "ja" } else { "nein" }
}
