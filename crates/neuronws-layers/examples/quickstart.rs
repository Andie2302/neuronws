//! Einstieg: Ein kleines Netz lernt XOR.
//!
//! Start: `cargo run -p neuronws-layers --features libm --example quickstart`

use neuronws_layers::activation::{Sigmoid, Tanh};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::adam::Adam;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::datasets::logic;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::BinaryCrossEntropy;
use neuronws_layers::random::Rng;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::ForwardPath;

fn main() {
    let mut rng = Rng::new(1);

    // 1. Das Netz: 2 Eingänge → 4 verborgene Neuronen (Tanh) → 1 Ausgang (Sigmoid).
    //    Alle Größen stehen im Typ, nichts davon braucht den Heap.
    let mut net = DenseLayer::<2, 4>::xavier(&mut rng)
        .then(ActivationLayer::<4, _>::new(Tanh))
        .then(DenseLayer::<4, 1>::xavier(&mut rng))
        .then(ActivationLayer::<1, _>::new(Sigmoid));

    // 2. Wie gelernt wird: Fehlerfunktion, Optimierer und Batchgröße.
    let data = logic::as_vectors(&logic::XOR);
    let mut trainer = Trainer::new(BinaryCrossEntropy, Adam::for_model(&net, 0.1), 4);

    // 3. Lernen: 300 Runden über alle vier Beispiele.
    let before = trainer.evaluate(&net, &data);
    let after = trainer.fit(&mut net, &data, 300);
    println!("Fehler vorher {before:.3}, nachher {after:.3}");

    // 4. Benutzen.
    for (input, target) in &data {
        let output = net.forward(input)[0];
        println!("{input:?} -> {output:.2} (soll {})", target[0]);
    }
}
