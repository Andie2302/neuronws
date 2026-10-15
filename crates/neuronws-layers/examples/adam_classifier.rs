//! Ein Netz aus austauschbaren Teilen: mehrere Schichten, Softmax, AdamW, Lernraten-Plan.
//!
//! Drei Punktwolken im Plan sollen drei Klassen ergeben. Alles, was hier gewählt wird
//! (Rechnung, Startwerte, Aktivierung, Fehlerfunktion, Optimierer, Batchgröße, Plan), ist ein
//! eigener, austauschbarer Baustein. Beim Kompilieren steht alles fest.
//!
//! Start: `cargo run -p neuronws-layers --features libm --example adam_classifier`

use neuronws_layers::activation::{Relu, Tanh};
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::adam::AdamW;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::Linear;
use neuronws_layers::init::{He, Xavier};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::neurons::neurons;
use neuronws_layers::optimizer::HasLearningRate;
use neuronws_layers::random::Rng;
use neuronws_layers::schedule::{Schedule, StepDecay};
use neuronws_layers::softmax_layer::Softmax;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::ForwardPath;

const CENTERS: [[f32; 2]; 3] = [[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]];

/// `per_class` Punkte um jedes Zentrum, Ziel ist der One-Hot-Vektor der Klasse.
fn make_data(rng: &mut Rng, per_class: usize) -> Vec<([f32; 2], [f32; 3])> {
    let mut data = Vec::new();
    for (class, [cx, cy]) in CENTERS.iter().enumerate() {
        for _ in 0..per_class {
            let point = [cx + rng.uniform(-1.5, 1.5), cy + rng.uniform(-1.5, 1.5)];
            let mut target = [0.0; 3];
            target[class] = 1.0;
            data.push((point, target));
        }
    }
    rng.shuffle(&mut data);
    data
}

fn argmax(values: &[f32; 3]) -> usize {
    (1..3).fold(0, |best, i| if values[i] > values[best] { i } else { best })
}

fn main() {
    let mut rng = Rng::new(42);
    let train = make_data(&mut rng, 40);
    let test = make_data(&mut rng, 100);

    // Drei Schichten mit Neuronen, jede mit eigener Rechnung, eigenen Startwerten und eigener
    // Aktivierung. Die letzte nimmt Softmax, damit die Ausgaben Wahrscheinlichkeiten sind.
    let mut net = neurons(Linear::<2, 8>::new(), &He, Elementwise::new(Relu), &mut rng)
        .then(neurons(
            Linear::<8, 8>::new(),
            &He,
            Elementwise::new(Tanh),
            &mut rng,
        ))
        .then(neurons(
            Linear::<8, 3>::new(),
            &Xavier,
            Softmax::<3>::new(),
            &mut rng,
        ));

    // Wie gelernt wird: Fehlerfunktion, Optimierer für das GANZE Netz, Mini-Batches zu 16.
    let mut trainer = Trainer::new(
        CategoricalCrossEntropy,
        AdamW::for_model(&net, 0.02, 0.01),
        16,
    );
    // Die Lernrate halbiert sich alle 25 Runden.
    let schedule = StepDecay::new(0.5, 25);
    let base = trainer.optimizer().learning_rate();

    println!("Runde   Lernrate   Verlust (Training)   Verlust (Test)");
    for epoch in 0..100 {
        trainer
            .optimizer_mut()
            .set_learning_rate(schedule.rate(epoch, base));
        let loss = trainer.epoch(&mut net, &train);
        if epoch % 10 == 0 || epoch == 99 {
            let rate = trainer.optimizer().learning_rate();
            let test_loss = trainer.evaluate(&net, &test);
            println!("{epoch:5}   {rate:8.5}   {loss:18.4}   {test_loss:14.4}");
        }
    }

    let correct = test
        .iter()
        .filter(|(x, target)| argmax(&net.forward(x)) == argmax(target))
        .count();
    println!(
        "\nRichtig erkannt auf neuen Daten: {correct} von {} ({:.1} %)",
        test.len(),
        100.0 * correct as f32 / test.len() as f32
    );
}
