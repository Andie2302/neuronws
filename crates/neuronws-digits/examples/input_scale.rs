//! Was passiert, wenn die Eingaben nicht 0 und 1 sind, sondern 0 und 100?
//!
//! Dasselbe Netz, dieselben Daten, dieselben Startwerte, nur die Pixel werden vor dem Netz
//! mit einem Faktor multipliziert. Vor jeder Tanh-Aktivierung sitzt ein Messfühler, der aufschreibt,
//! wie groß die eingehenden Werte `z` sind und wie groß die Ableitung dort ist.
//!
//! Start: `cargo run --release -p neuronws-digits --example input_scale`

use std::cell::Cell;

use neuronws_digits::{CLASSES, HIDDEN, PIXELS, Sample, accuracy, noisy_dataset};
use neuronws_layers::activation::{Activation, Tanh};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::scalar::Scalar;
use neuronws_layers::softmax_layer::SoftmaxLayer;
use neuronws_layers::training::train_epoch;
use neuronws_layers::traits::ForwardPath;

/// Die Messwerte: alle `z`, die in die Aktivierung hineingegangen sind.
#[derive(Debug, Default)]
struct Stats {
    count: Cell<usize>,
    sum_abs_z: Cell<f64>,
    saturated: Cell<usize>,
    sum_derivative: Cell<f64>,
}

impl Stats {
    fn reset(&self) {
        self.count.set(0);
        self.sum_abs_z.set(0.0);
        self.saturated.set(0);
        self.sum_derivative.set(0.0);
    }

    fn report(&self) -> String {
        let n = self.count.get() as f64;
        format!(
            "mittleres |z| = {:7.2}, gesättigt (|z| > 3): {:5.1} %, mittlere Ableitung: {:.4}",
            self.sum_abs_z.get() / n,
            100.0 * self.saturated.get() as f64 / n,
            self.sum_derivative.get() / n,
        )
    }
}

/// Eine Tanh-Aktivierung, die nebenbei aufschreibt, was in sie hineingeht.
#[derive(Debug)]
struct MeasuringTanh<'a>(&'a Stats);

impl Activation for MeasuringTanh<'_> {
    fn apply<T: Scalar>(&self, z: T) -> T {
        let stats = self.0;
        let z64 = z.to_f64();
        stats.count.set(stats.count.get() + 1);
        stats.sum_abs_z.set(stats.sum_abs_z.get() + z64.abs());
        stats
            .sum_derivative
            .set(stats.sum_derivative.get() + (1.0 - z64.tanh().powi(2)));
        if z64.abs() > 3.0 {
            stats.saturated.set(stats.saturated.get() + 1);
        }
        Tanh.apply(z)
    }

    fn derivative<T: Scalar>(&self, z: T) -> T {
        Tanh.derivative(z)
    }
}

fn scaled(data: &[Sample], factor: f32) -> Vec<Sample> {
    data.iter()
        .map(|(image, label)| (image.map(|pixel| pixel * factor), *label))
        .collect()
}

fn experiment(title: &str, factor: f32, learning_rate: f32) {
    println!("\n=== {title}: Pixel 0 und {factor}, Lernrate {learning_rate} ===");
    let mut rng = Rng::new(2026);
    let mut training = scaled(&noisy_dataset::<300>(2, &mut rng), factor);
    let test = scaled(&noisy_dataset::<200>(2, &mut rng), factor);

    let stats = Stats::default();
    let mut net = DenseLayer::<PIXELS, HIDDEN>::xavier(&mut rng)
        .then(ActivationLayer::<HIDDEN, _>::new(MeasuringTanh(&stats)))
        .then(DenseLayer::<HIDDEN, CLASSES>::xavier(&mut rng))
        .then(SoftmaxLayer::<CLASSES>::new());
    let mut sgd = Sgd::new(learning_rate);

    let measure = |net: &_| {
        stats.reset();
        for (image, _) in &test {
            let _: [f32; CLASSES] = ForwardPath::forward(net, image);
        }
        stats.report()
    };

    println!("vor dem Lernen:   {}", measure(&net));
    println!(
        "                  Test {:5.1} % richtig",
        100.0 * accuracy(&net, &test)
    );
    for epoch in 1..=60 {
        rng.shuffle(&mut training);
        let loss = train_epoch(&mut net, &mut sgd, &CategoricalCrossEntropy, &training, 10);
        if [1, 5, 20, 60].contains(&epoch) {
            println!(
                "Epoche {epoch:2}: Verlust {loss:8.4}, Test {:5.1} % richtig",
                100.0 * accuracy(&net, &test)
            );
        }
    }
    println!("nach dem Lernen:  {}", measure(&net));
}

fn main() {
    experiment("Normal", 1.0, 0.1);
    experiment("Große Eingaben", 100.0, 0.1);
    experiment("Große Eingaben, viel kleinere Lernrate", 100.0, 0.001);
    experiment("Große Eingaben, noch kleinere Lernrate", 100.0, 0.00001);
}
