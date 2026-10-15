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
use neuronws_layers::conv::Conv2d;
use neuronws_layers::dense_layer::Linear;
use neuronws_layers::init::{He, Xavier};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::neurons::neurons;
use neuronws_layers::pooling::MaxPool2d;
use neuronws_layers::random::Rng;
use neuronws_layers::reshape::Reshape;
use neuronws_layers::softmax_layer::Softmax;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{ForwardPath, Trainable};
use neuronws_layers::values::Image;

const CLASSES: [&str; 3] = ["senkrecht", "waagrecht", "diagonal"];

type Sample = (Image<1, 6, 6>, [f32; 3]);

/// Ein Bild der Klasse `class` mit der Linie an zufälliger Stelle und Rauschen.
fn make_image(rng: &mut Rng, class: usize) -> Sample {
    let mut image = [[[0.0_f32; 6]; 6]];
    let at = rng.next_below(6);
    for i in 0..6 {
        match class {
            0 => image[0][i][at] = 1.0,
            1 => image[0][at][i] = 1.0,
            _ if at.is_multiple_of(2) => image[0][i][i] = 1.0,
            _ => image[0][i][5 - i] = 1.0,
        }
    }
    for value in image[0].iter_mut().flatten() {
        *value += rng.uniform(-0.2, 0.2);
    }
    let mut target = [0.0; 3];
    target[class] = 1.0;
    (image, target)
}

fn make_data(rng: &mut Rng, per_class: usize) -> Vec<Sample> {
    let mut data: Vec<Sample> = (0..per_class * 3).map(|i| make_image(rng, i % 3)).collect();
    rng.shuffle(&mut data);
    data
}

/// Anzahl der lernbaren Parameter eines beliebigen Netzes, ohne den Typ hinzuschreiben.
fn parameter_count<M: Trainable>(_model: &M) -> usize {
    M::PARAMETER_COUNT
}

fn argmax(values: &[f32; 3]) -> usize {
    (1..3).fold(0, |best, i| if values[i] > values[best] { i } else { best })
}

fn main() {
    let mut rng = Rng::new(7);
    let train = make_data(&mut rng, 40);
    let test = make_data(&mut rng, 100);

    let mut net = Block::initialized(Conv2d::<1, 4, 6, 6, 3, 3, 4, 4>::new(), &He, &mut rng)
        .then(Block::<Elementwise<Image<4, 4, 4>, Relu>>::new(Relu))
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
            let hits = test
                .iter()
                .filter(|(x, t)| argmax(&net.forward(x)) == argmax(t))
                .count();
            println!(
                "{epoch:5}   {loss:18.4}   {test_loss:14.4}   {hits:3} von {}",
                test.len()
            );
        }
    }

    println!("\nPro Klasse (neue Bilder):");
    for (class, name) in CLASSES.iter().enumerate() {
        let of_class: Vec<_> = test.iter().filter(|(_, t)| argmax(t) == class).collect();
        let hits = of_class
            .iter()
            .filter(|(x, _)| argmax(&net.forward(x)) == class)
            .count();
        println!("  {name:10} {hits:3} von {}", of_class.len());
    }
}
