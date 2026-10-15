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
use neuronws_layers::datasets::blobs;
use neuronws_layers::dense_layer::Linear;
use neuronws_layers::init::{He, Xavier};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::metrics::accuracy;
use neuronws_layers::neurons::neurons;
use neuronws_layers::optimizer::HasLearningRate;
use neuronws_layers::random::Rng;
use neuronws_layers::schedule::{Schedule, StepDecay};
use neuronws_layers::softmax_layer::Softmax;
use neuronws_layers::trainer::Trainer;

fn main() {
    let mut rng = Rng::new(42);
    let train = blobs::<120>(&mut rng);
    let test = blobs::<300>(&mut rng);

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

    let hits: f32 = accuracy(&net, &test);
    println!(
        "\nRichtig erkannt auf neuen Daten: {:.1} % ({} Beispiele)",
        100.0 * hits,
        test.len()
    );
}
