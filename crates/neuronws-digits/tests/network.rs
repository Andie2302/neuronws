//! Das Netz braucht Mathe-Funktionen, also `std` oder `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_digits::{CLASSES, DIGITS, accuracy, new_network, noisy_dataset, one_hot};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
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
