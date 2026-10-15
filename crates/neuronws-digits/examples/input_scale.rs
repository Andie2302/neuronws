//! Wie wichtig ist die Größe der Eingaben? Dasselbe Netz, dieselben Bilder, dieselben Startwerte,
//! nur die Pixel werden vor dem Netz mit einem Faktor multipliziert.
//!
//! Vor jeder Aktivierung sitzt ein Messfühler. Er schreibt auf, wie groß die eingehenden Werte `z`
//! sind und bei wie vielen die Ableitung kleiner als 0,05 ist, also kaum noch ein Lernsignal ankommt.
//!
//! Start: `cargo run --release -p neuronws-digits --example input_scale`

use std::cell::Cell;

use neuronws_digits::{CLASSES, HIDDEN, Image, Label, PIXELS, Sample, accuracy, noisy_dataset};
use neuronws_layers::activation::{Activation, Relu, Tanh};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::scalar::Scalar;
use neuronws_layers::softmax_layer::SoftmaxLayer;
use neuronws_layers::standardize::Standardize;
use neuronws_layers::training::train_epoch;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

/// Die Messwerte: alle `z`, die in die Aktivierung hineingegangen sind.
#[derive(Debug, Default)]
struct Stats {
    count: Cell<usize>,
    sum_abs_z: Cell<f64>,
    no_signal: Cell<usize>,
}

impl Stats {
    fn reset(&self) {
        self.count.set(0);
        self.sum_abs_z.set(0.0);
        self.no_signal.set(0);
    }

    fn mean_abs_z(&self) -> f64 {
        self.sum_abs_z.get() / self.count.get() as f64
    }

    fn percent_without_signal(&self) -> f64 {
        100.0 * self.no_signal.get() as f64 / self.count.get() as f64
    }
}

/// Eine Aktivierung, die nebenbei aufschreibt, was in sie hineingeht.
#[derive(Debug)]
struct Measuring<'a, A>(&'a Stats, A);

impl<A: Activation> Activation for Measuring<'_, A> {
    fn apply<T: Scalar>(&self, z: T) -> T {
        let stats = self.0;
        stats.count.set(stats.count.get() + 1);
        stats
            .sum_abs_z
            .set(stats.sum_abs_z.get() + z.to_f64().abs());
        if self.1.derivative(z).to_f64() < 0.05 {
            stats.no_signal.set(stats.no_signal.get() + 1);
        }
        self.1.apply(z)
    }

    fn derivative<T: Scalar>(&self, z: T) -> T {
        self.1.derivative(z)
    }
}

fn scaled(data: &[Sample], factor: f32) -> Vec<Sample> {
    data.iter()
        .map(|(image, label)| (image.map(|pixel| pixel * factor), *label))
        .collect()
}

struct Outcome {
    mean_abs_z_start: f64,
    no_signal_start: f64,
    accuracy_after: f32,
    loss_after: f32,
    mean_abs_z_end: f64,
}

/// Lernt 60 Epochen und misst vorher und nachher. Gilt für jedes Netz mit dieser Ein- und Ausgabe.
fn evaluate<M>(
    mut net: M,
    stats: &Stats,
    rng: &mut Rng,
    mut training: Vec<Sample>,
    test: &[Sample],
    learning_rate: f32,
) -> Outcome
where
    M: ForwardPath<ForwardInput = Image, ForwardOutput = Label>
        + ForwardPathMut<ForwardInput = Image, ForwardOutput = Label>
        + BackwardPathMut<BackwardInput = Label, BackwardOutput = Image>
        + Trainable<Scalar = f32>,
{
    let mut sgd = Sgd::new(learning_rate);
    let measure = |net: &M| {
        stats.reset();
        for (image, _) in test {
            let _: Label = ForwardPath::forward(net, image);
        }
    };

    measure(&net);
    let (mean_abs_z_start, no_signal_start) = (stats.mean_abs_z(), stats.percent_without_signal());

    let mut loss_after = 0.0;
    for _ in 0..60 {
        rng.shuffle(&mut training);
        loss_after = train_epoch(&mut net, &mut sgd, &CategoricalCrossEntropy, &training, 10);
    }

    measure(&net);
    Outcome {
        mean_abs_z_start,
        no_signal_start,
        accuracy_after: accuracy(&net, test),
        loss_after,
        mean_abs_z_end: stats.mean_abs_z(),
    }
}

fn data(factor: f32) -> (Rng, Vec<Sample>, Vec<Sample>) {
    let mut rng = Rng::new(2026);
    let training = scaled(&noisy_dataset::<300>(2, &mut rng), factor);
    let test = scaled(&noisy_dataset::<200>(2, &mut rng), factor);
    (rng, training, test)
}

/// Das Netz direkt auf den (skalierten) Pixeln.
fn experiment<A: Activation>(activation: A, factor: f32, learning_rate: f32) -> Outcome {
    let (mut rng, training, test) = data(factor);
    let stats = Stats::default();
    let net = DenseLayer::<PIXELS, HIDDEN>::xavier(&mut rng)
        .then(ActivationLayer::<HIDDEN, _>::new(Measuring(
            &stats, activation,
        )))
        .then(DenseLayer::<HIDDEN, CLASSES>::xavier(&mut rng))
        .then(SoftmaxLayer::<CLASSES>::new());
    evaluate(net, &stats, &mut rng, training, &test, learning_rate)
}

/// Dasselbe Netz, aber mit einer `Standardize`-Schicht davor, die aus den Lerndaten berechnet wird.
fn experiment_standardized<A: Activation>(
    activation: A,
    factor: f32,
    learning_rate: f32,
) -> Outcome {
    let (mut rng, training, test) = data(factor);
    let stats = Stats::default();
    let standardize = Standardize::fit(training.iter().map(|(image, _)| image));
    let net = standardize
        .then(DenseLayer::<PIXELS, HIDDEN>::xavier(&mut rng))
        .then(ActivationLayer::<HIDDEN, _>::new(Measuring(
            &stats, activation,
        )))
        .then(DenseLayer::<HIDDEN, CLASSES>::xavier(&mut rng))
        .then(SoftmaxLayer::<CLASSES>::new());
    evaluate(net, &stats, &mut rng, training, &test, learning_rate)
}

fn print_header(title: &str) {
    println!("\n{title}");
    println!(
        "  Aktivierung  Faktor  Lernrate | Start: mittl. |z|  ohne Signal | nach 60 Epochen: Treffer  Verlust  mittl. |z|"
    );
}

fn print_row(name: &str, factor: f32, learning_rate: f32, r: &Outcome) {
    println!(
        "  {name:<11} {factor:>7} {learning_rate:>9} | {:>16.2} {:>10.1} % | {:>22.1} % {:>8.3} {:>9.2}",
        r.mean_abs_z_start,
        r.no_signal_start,
        100.0 * r.accuracy_after,
        r.loss_after,
        r.mean_abs_z_end,
    );
}

fn main() {
    print_header("1. Dieselbe Lernrate (0,1), drei Größen der Eingaben, zwei Aktivierungen");
    for factor in [1.0, 100.0, 0.01] {
        print_row("Tanh", factor, 0.1, &experiment(Tanh, factor, 0.1));
        print_row("ReLU", factor, 0.1, &experiment(Relu, factor, 0.1));
    }

    print_header(
        "2. Sehr kleine Eingaben (0 und 0,01): lässt sich das mit der Lernrate ausgleichen?",
    );
    for learning_rate in [0.1, 1.0, 10.0, 100.0, 1000.0] {
        print_row(
            "Tanh",
            0.01,
            learning_rate,
            &experiment(Tanh, 0.01, learning_rate),
        );
        print_row(
            "ReLU",
            0.01,
            learning_rate,
            &experiment(Relu, 0.01, learning_rate),
        );
    }

    print_header("3. Große Eingaben (0 und 100): lässt sich das mit der Lernrate ausgleichen?");
    for learning_rate in [0.1, 0.001, 0.00001, 0.0000001] {
        print_row(
            "Tanh",
            100.0,
            learning_rate,
            &experiment(Tanh, 100.0, learning_rate),
        );
        print_row(
            "ReLU",
            100.0,
            learning_rate,
            &experiment(Relu, 100.0, learning_rate),
        );
    }

    print_header(
        "4. Mit Standardize vor dem Netz (aus den Lerndaten berechnet), Lernrate 0,1: die Größe der Eingaben spielt keine Rolle mehr",
    );
    for factor in [1.0, 100.0, 0.01] {
        print_row(
            "Tanh",
            factor,
            0.1,
            &experiment_standardized(Tanh, factor, 0.1),
        );
        print_row(
            "ReLU",
            factor,
            0.1,
            &experiment_standardized(Relu, factor, 0.1),
        );
    }
}
