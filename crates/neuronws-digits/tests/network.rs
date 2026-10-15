//! Das Netz braucht Mathe-Funktionen, also `std` oder `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_digits::{
    CLASSES, DIGITS, HIDDEN, PIXELS, Sample, accuracy, new_network, noisy_dataset, one_hot,
};
use neuronws_layers::activation::Tanh;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::softmax_layer::SoftmaxLayer;
use neuronws_layers::standardize::Standardize;
use neuronws_layers::training::train_epoch;

#[test]
fn an_untrained_network_is_no_better_than_guessing() {
    let mut rng = Rng::new(3);
    let net = new_network(&mut rng);
    let test = noisy_dataset::<200>(2, &mut rng);
    assert!(accuracy(&net, &test) < 0.3);
}

#[test]
fn the_network_learns_to_recognise_digits_it_has_never_seen() {
    let mut rng = Rng::new(2026);
    let mut training = noisy_dataset::<300>(2, &mut rng);
    let test = noisy_dataset::<200>(2, &mut rng);
    let mut net = new_network(&mut rng);
    let mut sgd = Sgd::new(0.1);

    for _ in 0..30 {
        rng.shuffle(&mut training);
        train_epoch(&mut net, &mut sgd, &CategoricalCrossEntropy, &training, 10);
    }

    assert!(
        accuracy(&net, &test) >= 0.97,
        "Test: {}",
        accuracy(&net, &test)
    );

    let clean: [_; CLASSES] = core::array::from_fn(|d| (DIGITS[d], one_hot(d)));
    assert_eq!(accuracy(&net, &clean), 1.0);

    // Mit mehr Rauschen als beim Lernen wird es schlechter, bleibt aber brauchbar.
    let noisier = noisy_dataset::<200>(4, &mut rng);
    assert!(
        accuracy(&net, &noisier) >= 0.9,
        "mit Rauschen: {}",
        accuracy(&net, &noisier)
    );
}

fn scaled(data: &[Sample], factor: f32) -> Vec<Sample> {
    data.iter()
        .map(|(image, label)| (image.map(|pixel| pixel * factor), *label))
        .collect()
}

/// Trainiert ein Netz mit `Standardize` vorn auf Bildern, deren Pixel 0 und `factor` sind.
fn accuracy_with_standardize(factor: f32) -> f32 {
    let mut rng = Rng::new(2026);
    let mut training = scaled(&noisy_dataset::<300>(2, &mut rng), factor);
    let test = scaled(&noisy_dataset::<200>(2, &mut rng), factor);

    let mut net = Standardize::fit(training.iter().map(|(image, _)| image))
        .then(DenseLayer::<PIXELS, HIDDEN>::xavier(&mut rng))
        .then(ActivationLayer::<HIDDEN, _>::new(Tanh))
        .then(DenseLayer::<HIDDEN, CLASSES>::xavier(&mut rng))
        .then(SoftmaxLayer::<CLASSES>::new());
    let mut sgd = Sgd::new(0.1);
    for _ in 0..30 {
        rng.shuffle(&mut training);
        train_epoch(&mut net, &mut sgd, &CategoricalCrossEntropy, &training, 10);
    }
    accuracy(&net, &test)
}

#[test]
fn with_standardize_the_size_of_the_pixel_values_does_not_matter() {
    let results = [0.01, 1.0, 100.0].map(accuracy_with_standardize);
    for accuracy in results {
        assert!(accuracy >= 0.95, "Treffer {accuracy}");
    }
    // Die Skalierung wird vollständig herausgerechnet, bis auf Rundungsfehler im Bereich 1e-6.
    assert!((results[0] - results[1]).abs() < 0.02 && (results[1] - results[2]).abs() < 0.02);
}

#[test]
fn without_standardize_large_pixel_values_break_learning() {
    let mut rng = Rng::new(2026);
    let mut training = scaled(&noisy_dataset::<300>(2, &mut rng), 100.0);
    let test = scaled(&noisy_dataset::<200>(2, &mut rng), 100.0);
    let mut net = new_network(&mut rng);
    let mut sgd = Sgd::new(0.1);
    for _ in 0..30 {
        rng.shuffle(&mut training);
        train_epoch(&mut net, &mut sgd, &CategoricalCrossEntropy, &training, 10);
    }
    assert!(
        accuracy(&net, &test) < 0.6,
        "Treffer {}",
        accuracy(&net, &test)
    );
}
