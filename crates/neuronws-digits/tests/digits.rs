use neuronws_digits::{
    CLASSES, DIGITS, PIXELS, accuracy, argmax, flip_pixels, new_network, noisy_dataset, one_hot,
};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::training::train_epoch;

fn differing_pixels(a: &[f32; PIXELS], b: &[f32; PIXELS]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[test]
fn the_ten_glyphs_are_all_different() {
    for (a, image_a) in DIGITS.iter().enumerate() {
        for (b, image_b) in DIGITS.iter().enumerate().skip(a + 1) {
            assert!(
                differing_pixels(image_a, image_b) >= 3,
                "{a} und {b} sehen zu ähnlich aus"
            );
        }
    }
}

#[test]
fn glyphs_are_parsed_from_the_text() {
    // Die Eins hat oben in der Mitte einen Pixel und unten eine Linie aus 3.
    let one = &DIGITS[1];
    assert_eq!(one[2], 1.0);
    assert_eq!(one[0], 0.0);
    assert_eq!(&one[6 * 5..], &[0.0, 1.0, 1.0, 1.0, 0.0]);
    assert!(DIGITS.iter().flatten().all(|&p| p == 0.0 || p == 1.0));
}

#[test]
fn one_hot_and_argmax_are_inverse() {
    for digit in 0..CLASSES {
        assert_eq!(argmax(&one_hot(digit)), digit);
        assert_eq!(one_hot(digit).iter().sum::<f32>(), 1.0);
    }
}

#[test]
fn flip_pixels_flips_exactly_the_requested_number() {
    let mut rng = Rng::new(1);
    for count in [0, 1, 5, PIXELS] {
        let flipped = flip_pixels(&DIGITS[8], count, &mut rng);
        assert_eq!(differing_pixels(&DIGITS[8], &flipped), count);
    }
}

#[test]
fn noise_is_reproducible_per_seed() {
    let a = noisy_dataset::<20>(3, &mut Rng::new(9));
    let b = noisy_dataset::<20>(3, &mut Rng::new(9));
    let c = noisy_dataset::<20>(3, &mut Rng::new(10));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn the_dataset_cycles_through_the_digits_and_limits_the_noise() {
    let data = noisy_dataset::<30>(2, &mut Rng::new(4));
    for (i, (image, label)) in data.iter().enumerate() {
        assert_eq!(argmax(label), i % CLASSES);
        assert!(differing_pixels(image, &DIGITS[i % CLASSES]) <= 2);
    }
}

#[test]
#[should_panic(expected = "durch 10 teilbar")]
fn dataset_size_must_be_a_multiple_of_ten() {
    let _ = noisy_dataset::<15>(1, &mut Rng::new(0));
}

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
