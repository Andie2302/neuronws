//! Ein kleines Faltungsnetz (CNN): Linien in verrauschten Bildern erkennen.
//!
//! Drei Klassen von 6×6-Bildern: senkrechte, waagrechte und diagonale Linie. Die Linie liegt an
//! zufälliger Stelle, dazu kommt Rauschen. Weil die Kerne der Faltung überall im Bild gelten,
//! muss das Netz nicht für jede Position einzeln lernen.
//!
//! Aufbau (Formen als Kanäle × Zeilen × Spalten):
//!
//! ```text
//! Bild 1×6×6 → Conv 3×3, 4 Kerne → 4×4×4 → ReLU → MaxPool 2×2 → 4×2×2 → flach 16 → Dense → Softmax 3
//! ```
//!
//! Start: `cargo run -p neuronws-layers --features libm --release --example cnn_lines`

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::adam::AdamW;
use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::conv::Conv2dValid;
use neuronws_layers::datasets::{LINE_CLASSES, lines};
use neuronws_layers::dense_layer::Linear;
use neuronws_layers::init::{He, Xavier};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::metrics::{accuracy, argmax};
use neuronws_layers::neurons::neurons;
use neuronws_layers::pooling::MaxPool2d;
use neuronws_layers::random::Rng;
use neuronws_layers::reshape::Reshape;
use neuronws_layers::softmax_layer::Softmax;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{ForwardPath, Trainable};
use neuronws_layers::values::Image;

/// Anzahl der lernbaren Parameter eines beliebigen Netzes, ohne den Typ hinzuschreiben.
fn parameter_count<M: Trainable>(_model: &M) -> usize {
    M::PARAMETER_COUNT
}

fn main() {
    let mut rng = Rng::new(7);
    let train = lines::<120>(&mut rng);
    let test = lines::<300>(&mut rng);

    // Faltung und ReLU in einem Aufruf: `neurons` kennt keine Formen und geht deshalb auch für Bilder.
    let mut net = neurons(
        Conv2dValid::<1, 4, 6, 6, 3, 3, 4, 4>::new(),
        &He,
        Elementwise::new(Relu),
        &mut rng,
    )
    .then(Block::from_parts(MaxPool2d::<4, 4, 4, 2, 2, 2>::new(), ()))
    .then(Block::from_parts(
        Reshape::<Image<4, 2, 2>, [f32; 16]>::new(),
        (),
    ))
    .then(neurons(
        Linear::<16, 3>::new(),
        &Xavier,
        Softmax::<3>::new(),
        &mut rng,
    ));

    println!(
        "Parameter: {} (Faltung 4·3·3 + 4 = 40, Dense 16·3 + 3 = 51)\n",
        parameter_count(&net)
    );

    let mut trainer = Trainer::new(
        CategoricalCrossEntropy,
        AdamW::for_model(&net, 0.01, 0.001),
        8,
    );

    println!("Runde   Verlust (Training)   Verlust (Test)   Treffer (Test)");
    for epoch in 0..40 {
        let loss = trainer.epoch(&mut net, &train);
        if epoch % 5 == 0 || epoch == 39 {
            let test_loss = trainer.evaluate(&net, &test);
            let hits = accuracy::<f32, _, _, _>(&net, &test);
            println!(
                "{epoch:5}   {loss:18.4}   {test_loss:14.4}   {:5.1} %",
                100.0 * hits
            );
        }
    }

    println!("\nPro Klasse (neue Bilder):");
    for (class, name) in LINE_CLASSES.iter().enumerate() {
        let of_class: Vec<_> = test.iter().filter(|(_, t)| argmax(t) == class).collect();
        let hits = of_class
            .iter()
            .filter(|(x, _)| argmax(&net.forward(x)) == class)
            .count();
        println!("  {name:10} {hits:3} von {}", of_class.len());
    }
}
