//! Adam und AdamW. Braucht `sqrt`, also das Feature `std` oder `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::Tanh;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::adam::{Adam, AdamW};
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::optimizer::{Momentum, Optimizer, Sgd};
use neuronws_layers::random::Rng;
use neuronws_layers::training::{mean_loss, train_epoch};
use neuronws_layers::traits::{BackwardPathMut, ForwardPathMut};

type Line = DenseLayer<1, 1, f64>;

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} gegen {b}");
}

/// Gradient: `grad_w = 2`, `grad_b = 1`.
fn one_gradient_step(layer: &mut Line) {
    layer.forward_mut(&[2.0]);
    layer.backward_mut(&[1.0]);
}

/// Mit Python in doppelter Genauigkeit unabhängig nachgerechnet (η = 0.1, β₁ = 0.9, β₂ = 0.999).
#[test]
fn adam_follows_the_formula_for_three_steps() {
    let mut layer = Line::new([[1.0]], [0.0]);
    let mut adam = Adam::for_model(&layer, 0.1);
    let expected = [
        (0.9000000005, -0.09999999900000002),
        (0.8000000010000007, -0.19999999799999935),
        (0.7000000015000007, -0.29999999699999935),
    ];
    for (w, b) in expected {
        one_gradient_step(&mut layer);
        adam.step(&mut layer, 1);
        assert_close(layer.weights()[0][0], w);
        assert_close(layer.bias()[0], b);
    }
}

#[test]
fn adamw_decays_weights_next_to_the_adam_step() {
    let mut layer = Line::new([[1.0]], [0.0]);
    let mut adamw = AdamW::for_model(&layer, 0.1, 0.5);
    let expected = [
        (0.8500000005, -0.09999999900000002),
        (0.7075000009750008, -0.19499999804999935),
        (0.5721250014262508, -0.2852499971474993),
    ];
    for (w, b) in expected {
        one_gradient_step(&mut layer);
        adamw.step(&mut layer, 1);
        assert_close(layer.weights()[0][0], w);
        assert_close(layer.bias()[0], b);
    }
}

#[test]
fn adamw_without_weight_decay_is_adam() {
    let mut a = Line::new([[1.0]], [0.5]);
    let mut b = a.clone();
    let mut adam = Adam::for_model(&a, 0.05);
    let mut adamw = AdamW::for_model(&b, 0.05, 0.0);
    for _ in 0..5 {
        one_gradient_step(&mut a);
        one_gradient_step(&mut b);
        adam.step(&mut a, 1);
        adamw.step(&mut b, 1);
    }
    assert_eq!(a.weights(), b.weights());
    assert_eq!(a.bias(), b.bias());
}

/// Die Schrittgröße hängt kaum von der Größe des Gradienten ab, das ist der Witz von Adam.
#[test]
fn the_first_step_is_about_the_learning_rate_whatever_the_gradient_size() {
    for scale in [0.001, 1.0, 1000.0] {
        let mut layer = Line::new([[0.0]], [0.0]);
        layer.forward_mut(&[scale]);
        layer.backward_mut(&[1.0]);
        Adam::for_model(&layer, 0.01).step(&mut layer, 1);
        assert!((layer.weights()[0][0] + 0.01).abs() < 1e-6, "{scale}");
    }
}

/// Jeder Parameter hat sein eigenes Gedächtnis: der kleine Gradient wird nicht vom großen verdrängt.
#[test]
fn every_parameter_has_its_own_step_size() {
    let mut layer = DenseLayer::<2, 1, f64>::new([[0.0, 0.0]], [0.0]);
    layer.forward_mut(&[1000.0, 0.001]);
    layer.backward_mut(&[1.0]);
    Adam::for_model(&layer, 0.01).step(&mut layer, 1);
    let [[w1, w2]] = *layer.weights();
    assert!(
        (w1 + 0.01).abs() < 1e-6 && (w2 + 0.01).abs() < 1e-6,
        "{w1}, {w2}"
    );
}

#[test]
fn the_gradients_are_averaged_over_the_batch() {
    let mut once = Line::new([[1.0]], [0.0]);
    let mut twice = once.clone();
    one_gradient_step(&mut once);
    one_gradient_step(&mut twice);
    one_gradient_step(&mut twice); // Gradient 4 und 2, durch 2 geteilt wie oben
    Adam::for_model(&once, 0.1).step(&mut once, 1);
    Adam::for_model(&twice, 0.1).step(&mut twice, 2);
    assert_eq!(once.weights(), twice.weights());
}

#[test]
fn adam_clears_the_gradients() {
    let mut layer = Line::new([[1.0]], [0.0]);
    one_gradient_step(&mut layer);
    Adam::for_model(&layer, 0.1).step(&mut layer, 1);
    assert_eq!(layer.grad_weights(), &[[0.0]]);
    assert_eq!(layer.grad_bias(), &[0.0]);
}

/// Ohne Gradient und mit Weight Decay schrumpfen die Gewichte Richtung 0.
#[test]
fn weight_decay_shrinks_weights_without_any_gradient() {
    let mut layer = Line::new([[1.0]], [0.0]);
    let mut adamw = AdamW::for_model(&layer, 0.1, 0.5);
    for _ in 0..3 {
        adamw.step(&mut layer, 1);
    }
    assert_close(layer.weights()[0][0], 0.95_f64.powi(3));
}

#[test]
fn custom_betas_and_learning_rate_can_be_set() {
    let layer = Line::new([[1.0]], [0.0]);
    let mut adam = Adam::for_model(&layer, 0.1).with_betas(0.8, 0.9, 1e-6);
    assert_close(adam.learning_rate(), 0.1);
    adam.set_learning_rate(0.01);
    assert_close(adam.learning_rate(), 0.01);
}

/// Adam passt zu einem zusammengesetzten Netz, ohne dass irgendwo eine Zahl der Parameter steht.
#[test]
fn adam_trains_a_chained_network_on_xor() {
    let mut rng = Rng::new(7);
    let mut net = DenseLayer::<2, 4, f32>::xavier(&mut rng)
        .then(ActivationLayer::<4, _>::new(Tanh))
        .then(DenseLayer::<4, 1, f32>::xavier(&mut rng));
    let data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];
    let mut adam = Adam::for_model(&net, 0.05);
    for _ in 0..500 {
        train_epoch(&mut net, &mut adam, &MeanSquaredError, &data, 4);
    }
    assert!(mean_loss(&net, &MeanSquaredError, &data) < 1e-3);
}

/// Auf demselben Netz und mit denselben Daten: Adam kommt mit weniger Runden aus als SGD.
#[test]
fn adam_needs_fewer_rounds_than_sgd_on_xor() {
    fn rounds<O: Optimizer<Net>>(mut optimizer: O) -> usize {
        let mut rng = Rng::new(7);
        let mut net = DenseLayer::<2, 4, f32>::xavier(&mut rng)
            .then(ActivationLayer::<4, _>::new(Tanh))
            .then(DenseLayer::<4, 1, f32>::xavier(&mut rng));
        let data = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ];
        for round in 0..5000 {
            if mean_loss(&net, &MeanSquaredError, &data) < 1e-3 {
                return round;
            }
            train_epoch(&mut net, &mut optimizer, &MeanSquaredError, &data, 4);
        }
        5000
    }
    type Net = neuronws_layers::chain::Chain<
        neuronws_layers::chain::Chain<DenseLayer<2, 4>, ActivationLayer<4, Tanh>>,
        DenseLayer<4, 1>,
    >;
    let adam = rounds(Adam::<Net>::new(0.05));
    let sgd = rounds(Sgd::new(0.05));
    let momentum = rounds(Momentum::<Net>::new(0.05, 0.9));
    assert!(adam < sgd, "Adam {adam}, Sgd {sgd}, Momentum {momentum}");
    assert!(adam < 1000, "Adam {adam}");
}
