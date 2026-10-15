//! Ziffernerkennung mit Adam statt SGD: dasselbe Netz, dieselben Daten, nur der Optimierer ändert
//! sich. Verglichen wird mit dem Ergebnis von `recognize_digits` (SGD, Rate 0,1).
//!
//! Start: `cargo run --release -p neuronws-digits --example adam_vs_sgd`
//!
//! # Was verglichen wird
//!
//! Das Netz ist in beiden Fällen dasselbe (35 Pixel → 16 Neuronen mit Tanh → 10 Ziffern mit
//! Softmax, Kreuzentropie als Fehler), ebenso die Daten und die Startgewichte: Jeder Zufallsstart
//! `s` gibt allen Verfahren dieselben Bilder und dieselben Anfangsgewichte. Drei Verfahren treten an:
//!
//! - **SGD wie bisher:** Rate 0,1, so wie in `recognize_digits`.
//! - **SGD mit gewählter Rate:** Die beste Rate aus einer kleinen Auswahl.
//! - **Adam mit gewählter Rate:** Dasselbe für Adam.
//!
//! Die Rate wird mit *anderen* Zufallsstarts gewählt (100 bis 103) als die, mit denen danach
//! verglichen wird (1 bis 8). Sonst würde die Wahl an denselben Daten abgelesen, auf denen später
//! gemessen wird, und das Ergebnis wäre zu schön. Gewählt wird nach dem Verlust auf neuen Bildern,
//! nicht nach der Trefferquote: Die liegt schon nach wenigen Runden bei fast 100 % und
//! unterscheidet kaum noch etwas.
//!
//! Der Optimierer ist der einzige Teil, der sich ändert, weil er nicht im Netz steckt, sondern im
//! `Trainer`. Dieselbe `learn`-Funktion trainiert mit jedem Optimierer: Sie bekommt eine Funktion,
//! die ihn für das Netz baut.
//!
//! # Was dabei herauskommt
//!
//! ```text
//!                                         SGD 0.1 (bisher)   SGD 0.3   Adam 0.01
//! Runden, bis der Verlust unter 0,05 liegt        16             7          5
//! Verlust nach 60 Runden                        0.0172        0.0116     0.0104
//! richtig erkannt nach 60 Runden                 99.7 %        99.8 %     99.8 %
//! richtig bei 6 falschen Pixeln                  90.6 %        90.7 %     89.1 %
//! ```
//!
//! - **Die Lernrate war der Engpass, nicht der Optimierer.** Die bisherige Rate 0,1 war für SGD zu
//!   klein: Mit 0,3 braucht es nur noch 7 statt 16 Runden. Von SGD mit der besseren Rate zu Adam
//!   sind es nur noch 2 Runden weniger (7 auf 5). Wer die Lernrate nicht anpasst, vergleicht die
//!   Rate und nicht den Optimierer.
//! - **Adam ist am Ende etwas genauer** (Verlust 0,0104 gegen 0,0116), erkennt aber nicht mehr
//!   Ziffern richtig: Auf Bildern mit 0 bis 2 falschen Pixeln, so wie beim Lernen, sind alle drei
//!   bei knapp 100 %, und die Aufgabe ist damit gelöst.
//! - **Bei stärkerem Rauschen hilft Adam nicht.** Mit 6 falschen Pixeln, mehr als das Netz je
//!   gesehen hat, liegen alle drei um 90 % (Adam 89,1 %, SGD 90,7 %). Ob ein so kleiner
//!   Unterschied mehr als Zufall ist, sagen 8 Zufallsstarts nicht. Ein kleinerer Verlust auf
//!   bekannten Bildern verspricht jedenfalls kein besseres Verhalten auf unbekannten.
//! - Beim alten Beispiel `recognize_digits` (ein Start, Seed 2026) kam nach 60 Runden ein Verlust
//!   von 0,0153 und 99,5 % heraus. Das passt zur ersten Spalte (Mittel aus 8 Starts: 0,0172 und
//!   99,7 %).

use neuronws_digits::{
    CLASSES, DIGITS, Image, Label, Network, accuracy, flip_pixels, new_network, noisy_dataset,
    one_hot,
};
use neuronws_layers::adam::Adam;
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::random::Rng;
use neuronws_layers::trainer::Trainer;

const EPOCHS: usize = 60;
const BATCH: usize = 10;
/// Die Zufallsstarts, mit denen verglichen wird.
const STARTS: std::ops::RangeInclusive<u64> = 1..=8;
/// Die Zufallsstarts, mit denen die Lernrate gewählt wird.
const RATE_STARTS: std::ops::RangeInclusive<u64> = 100..=103;

/// Der Verlauf eines Trainings: nach jeder Runde Verlust und Trefferquote auf neuen Bildern.
struct Curve {
    loss: [f32; EPOCHS],
    accuracy: [f32; EPOCHS],
}

/// Lernt von einem Zufallsstart aus. `make_optimizer` baut den Optimierer für das Netz.
fn learn<O: Optimizer<Network>>(
    seed: u64,
    make_optimizer: impl Fn(&Network) -> O,
) -> (Network, Curve) {
    let mut rng = Rng::new(seed);
    // Lernen: 300 Bilder mit 0 bis 2 falschen Pixeln. Prüfen: 300 andere, genauso verrauschte.
    let mut training = noisy_dataset::<300>(2, &mut rng);
    let test = noisy_dataset::<300>(2, &mut rng);
    let mut net = new_network(&mut rng);
    let mut trainer = Trainer::new(CategoricalCrossEntropy, make_optimizer(&net), BATCH);

    let mut curve = Curve {
        loss: [0.0; EPOCHS],
        accuracy: [0.0; EPOCHS],
    };
    for epoch in 0..EPOCHS {
        let _training_loss: f32 = trainer.epoch_shuffled(&mut net, &mut training, &mut rng);
        curve.loss[epoch] = trainer.evaluate(&net, &test);
        curve.accuracy[epoch] = accuracy(&net, &test);
    }
    (net, curve)
}

/// Das Ergebnis eines Verfahrens über alle Zufallsstarts.
struct Study {
    /// Der Mittelwert der Verläufe.
    mean: Curve,
    /// Die fertigen Netze, eines pro Zufallsstart.
    nets: Vec<Network>,
}

fn study<O: Optimizer<Network>>(
    starts: std::ops::RangeInclusive<u64>,
    make_optimizer: impl Fn(&Network) -> O,
) -> Study {
    let mut mean = Curve {
        loss: [0.0; EPOCHS],
        accuracy: [0.0; EPOCHS],
    };
    let mut nets = Vec::new();
    let count = starts.clone().count() as f32;
    for seed in starts {
        let (net, curve) = learn(seed, &make_optimizer);
        for epoch in 0..EPOCHS {
            mean.loss[epoch] += curve.loss[epoch] / count;
            mean.accuracy[epoch] += curve.accuracy[epoch] / count;
        }
        nets.push(net);
    }
    Study { mean, nets }
}

/// Probiert die Raten durch und liefert die mit dem kleinsten Verlust nach der letzten Runde.
fn choose_rate<O: Optimizer<Network>>(
    name: &str,
    rates: &[f32],
    make_optimizer: impl Fn(&Network, f32) -> O,
) -> f32 {
    print!("  {name:5}");
    let mut best = (f32::INFINITY, rates[0]);
    for &rate in rates {
        let loss = study(RATE_STARTS, |net| make_optimizer(net, rate))
            .mean
            .loss[EPOCHS - 1];
        print!("  Rate {rate}: {loss:.4}");
        if loss < best.0 {
            best = (loss, rate);
        }
    }
    println!("   → gewählt: {}", best.1);
    best.1
}

/// Die erste Runde (ab 1 gezählt), in der der mittlere Verlust unter `limit` liegt.
fn epochs_until(curve: &Curve, limit: f32) -> Option<usize> {
    curve
        .loss
        .iter()
        .position(|&loss| loss < limit)
        .map(|i| i + 1)
}

/// Frisch verrauschte Bilder: je `count` Stück mit genau `flips` falschen Pixeln.
fn noisy_images(flips: usize, count: usize, rng: &mut Rng) -> Vec<(Image, Label)> {
    (0..count)
        .map(|i| {
            (
                flip_pixels(&DIGITS[i % CLASSES], flips, rng),
                one_hot(i % CLASSES),
            )
        })
        .collect()
}

fn main() {
    println!(
        "1. Lernrate wählen (Verlust auf neuen Bildern nach {EPOCHS} Runden, Mittel aus 4 Starts)"
    );
    let sgd_rate = choose_rate("SGD", &[0.03, 0.1, 0.3, 1.0], |net, rate| {
        Sgd::for_model(net, rate)
    });
    let adam_rate = choose_rate("Adam", &[0.001, 0.003, 0.01, 0.03], |net, rate| {
        Adam::for_model(net, rate)
    });

    let candidates = [
        (
            "SGD 0.1 (bisher)".to_string(),
            study(STARTS, |net| Sgd::for_model(net, 0.1)),
        ),
        (
            format!("SGD {sgd_rate}"),
            study(STARTS, |net| Sgd::for_model(net, sgd_rate)),
        ),
        (
            format!("Adam {adam_rate}"),
            study(STARTS, |net| Adam::for_model(net, adam_rate)),
        ),
    ];

    println!("\n2. Verlauf auf neuen Bildern (Mittel aus 8 Starts)");
    print!("  Runde |");
    for (name, _) in &candidates {
        print!(" {name:^19} |");
    }
    print!("\n        |");
    for _ in &candidates {
        print!(" {:^19} |", "Verlust  richtig");
    }
    println!();
    for epoch in [1, 2, 3, 5, 10, 20, 40, 60] {
        print!("  {epoch:5} |");
        for (_, study) in &candidates {
            let cell = format!(
                "{:.4}  {:5.1} %",
                study.mean.loss[epoch - 1],
                100.0 * study.mean.accuracy[epoch - 1]
            );
            print!(" {cell:^19} |");
        }
        println!();
    }
    println!("\n  Runden, bis der mittlere Verlust unter 0.05 liegt:");
    for (name, study) in &candidates {
        match epochs_until(&study.mean, 0.05) {
            Some(n) => println!("    {name:16} {n}"),
            None => println!("    {name:16} nicht erreicht"),
        }
    }

    println!(
        "\n3. Wie viele falsche Pixel verträgt das Netz? (je 500 Bilder, fertige Netze aller Starts)"
    );
    print!("  Falsche Pixel |");
    for (name, _) in &candidates {
        print!(" {name:^16} |");
    }
    println!();
    let mut rng = Rng::new(999);
    for flips in 0..=6 {
        let images = noisy_images(flips, 500, &mut rng);
        print!("  {flips:13} |");
        for (_, study) in &candidates {
            let mean: f32 = study
                .nets
                .iter()
                .map(|net| accuracy(net, &images))
                .sum::<f32>()
                / study.nets.len() as f32;
            let cell = format!("{:.1} %", 100.0 * mean);
            print!(" {cell:^16} |");
        }
        println!();
    }
}
