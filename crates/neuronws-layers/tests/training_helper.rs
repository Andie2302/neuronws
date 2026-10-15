use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::random::Rng;
use neuronws_layers::training::{mean_loss, train_epoch};
use neuronws_layers::traits::{BackwardPathMut, ForwardPathMut, Trainable};

type Line = DenseLayer<1, 1, f32>;
type LineData = [([f32; 1], [f32; 1]); 5];

fn line_data() -> LineData {
    [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x: f32| ([x], [2.0 * x + 1.0]))
}

fn assert_close(a: f32, b: f32, tolerance: f32) {
    assert!((a - b).abs() < tolerance, "{a} gegen {b}");
}

#[test]
fn full_batch_equals_the_hand_written_loop() {
    let data = line_data();

    let mut by_hand = Line::new([[0.0]], [0.0]);
    let mut sgd = Sgd::new(0.05);
    for _ in 0..50 {
        for (x, t) in &data {
            let y = *by_hand.forward_mut(x);
            by_hand.backward_mut(&MeanSquaredError.gradient(&y, t));
        }
        sgd.step(&mut by_hand, data.len());
    }

    let mut helper = Line::new([[0.0]], [0.0]);
    let mut sgd = Sgd::new(0.05);
    for _ in 0..50 {
        train_epoch(&mut helper, &mut sgd, &MeanSquaredError, &data, data.len());
    }

    assert_eq!(by_hand.weights(), helper.weights());
    assert_eq!(by_hand.bias(), helper.bias());
}

#[test]
fn every_batch_size_learns_the_line() {
    let data = line_data();
    for batch_size in [1, 2, 3, 5, 100] {
        let mut layer = Line::new([[0.0]], [0.0]);
        let mut sgd = Sgd::new(0.02);
        for _ in 0..1500 {
            train_epoch(&mut layer, &mut sgd, &MeanSquaredError, &data, batch_size);
        }
        assert_close(layer.weights()[0][0], 2.0, 1e-2);
        assert_close(layer.bias()[0], 1.0, 1e-2);
    }
}

/// Schreibt mit, mit welchen Gruppengrößen der Optimierer aufgerufen wird.
struct Recorder(Vec<usize>);

impl<M: Trainable> Optimizer<M> for Recorder {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        self.0.push(batch_size);
        model.zero_grad();
    }
}

#[test]
fn the_last_batch_is_smaller_and_is_averaged_with_its_real_size() {
    let mut layer = Line::new([[0.0]], [0.0]);
    let mut recorder = Recorder(Vec::new());
    train_epoch(
        &mut layer,
        &mut recorder,
        &MeanSquaredError,
        &line_data(),
        2,
    );
    assert_eq!(recorder.0, [2, 2, 1]);
}

#[test]
fn it_returns_the_mean_loss_of_the_epoch() {
    // Lernrate 0: Das Modell bleibt gleich, der Verlust ist leicht von Hand zu rechnen.
    let mut layer = Line::new([[0.0]], [0.0]);
    let data = line_data();
    let expected = data.iter().map(|(_, [t])| t * t).sum::<f32>() / 5.0;

    let mut sgd = Sgd::new(0.0);
    let loss = train_epoch(&mut layer, &mut sgd, &MeanSquaredError, &data, 2);

    assert_close(loss, expected, 1e-5);
    assert_close(mean_loss(&layer, &MeanSquaredError, &data), expected, 1e-5);
}

#[test]
fn the_loss_goes_down_while_learning() {
    let data = line_data();
    let mut layer = Line::new([[0.0]], [0.0]);
    let mut sgd = Sgd::new(0.05);

    let first = train_epoch(&mut layer, &mut sgd, &MeanSquaredError, &data, 5);
    let mut last = first;
    for _ in 0..100 {
        last = train_epoch(&mut layer, &mut sgd, &MeanSquaredError, &data, 5);
    }
    assert!(last < first / 1000.0, "{first} -> {last}");
}

#[test]
fn it_works_with_f64() {
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x: f64| ([x], [2.0 * x + 1.0]));
    let mut layer = DenseLayer::<1, 1, f64>::new([[0.0]], [0.0]);
    let mut sgd = Sgd::<f64>::new(0.05);
    for _ in 0..500 {
        train_epoch(&mut layer, &mut sgd, &MeanSquaredError, &data, 2);
    }
    assert!((layer.weights()[0][0] - 2.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "batch_size")]
fn batch_size_zero_panics() {
    let mut layer = Line::new([[0.0]], [0.0]);
    train_epoch(
        &mut layer,
        &mut Sgd::new(0.1),
        &MeanSquaredError,
        &line_data(),
        0,
    );
}

#[test]
#[should_panic(expected = "data darf nicht leer sein")]
fn empty_data_panics() {
    let mut layer = Line::new([[0.0]], [0.0]);
    let data: [([f32; 1], [f32; 1]); 0] = [];
    train_epoch(&mut layer, &mut Sgd::new(0.1), &MeanSquaredError, &data, 1);
}

#[test]
fn shuffle_keeps_all_elements() {
    let mut items: [u32; 20] = core::array::from_fn(|i| i as u32);
    Rng::new(1).shuffle(&mut items);
    let mut sorted = items;
    sorted.sort_unstable();
    assert_eq!(sorted, core::array::from_fn::<u32, 20, _>(|i| i as u32));
}

#[test]
fn shuffle_is_deterministic_per_seed_and_actually_reorders() {
    let original: [u32; 20] = core::array::from_fn(|i| i as u32);
    let shuffled = |seed| {
        let mut items = original;
        Rng::new(seed).shuffle(&mut items);
        items
    };
    assert_eq!(shuffled(7), shuffled(7));
    assert_ne!(shuffled(7), shuffled(8));
    assert_ne!(shuffled(7), original);
}

#[test]
fn shuffle_handles_tiny_slices() {
    let mut rng = Rng::new(3);
    let mut empty: [u8; 0] = [];
    rng.shuffle(&mut empty);
    let mut one = [42];
    rng.shuffle(&mut one);
    assert_eq!(one, [42]);
}

#[test]
fn next_below_stays_in_range_and_hits_every_value() {
    let mut rng = Rng::new(5);
    let mut seen = [false; 6];
    for _ in 0..1000 {
        let n = rng.next_below(6);
        assert!(n < 6);
        seen[n] = true;
    }
    assert!(seen.iter().all(|&s| s));
}
