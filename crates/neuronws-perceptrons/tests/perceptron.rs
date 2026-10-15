use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::datasets::logic::{AND, OR, XOR};
use neuronws_layers::traits::ForwardPath;
use neuronws_perceptrons::Perceptron;

fn assert_classifies(perceptron: &Perceptron<2>, data: &[([f32; 2], f32)]) {
    for (input, target) in data {
        assert_eq!(perceptron.predict(input), *target, "{input:?}");
    }
}

#[test]
fn predict_applies_the_threshold() {
    let and = Perceptron::<2>::new([1.0, 1.0], -1.5);
    assert_classifies(&and, &AND);
}

#[test]
fn learn_follows_the_perceptron_rule() {
    // y = Step(0) = 1, Ziel 0, Fehler = 1: w ← w − η x, b ← b − η
    let mut p = Perceptron::<2>::new([0.0, 0.0], 0.0);
    let error = p.learn(&[1.0, 1.0], 0.0, 0.5);

    assert_eq!(error, 1.0);
    assert_eq!(p.weights(), &[-0.5, -0.5]);
    assert_eq!(p.bias(), -0.5);
}

#[test]
fn learn_does_nothing_when_the_prediction_is_right() {
    let mut p = Perceptron::<2>::new([1.0, 1.0], -1.5);
    let error = p.learn(&[1.0, 1.0], 1.0, 0.5);

    assert_eq!(error, 0.0);
    assert_eq!(p.weights(), &[1.0, 1.0]);
    assert_eq!(p.bias(), -1.5);
}

#[test]
fn fit_learns_and_and_or() {
    for data in [&AND, &OR] {
        let mut p = Perceptron::<2>::new([0.0, 0.0], 0.0);
        assert!(p.fit(data, 100, 0.1).is_some());
        assert_classifies(&p, data);
    }
}

#[test]
fn fit_cannot_learn_xor() {
    // XOR ist nicht linear trennbar: Egal wie lange man lernt, es bleiben Fehler.
    let mut p = Perceptron::<2>::new([0.0, 0.0], 0.0);
    assert_eq!(p.fit(&XOR, 1000, 0.1), None);
}

#[test]
fn perceptron_chains_like_any_layer() {
    let and = Perceptron::<2>::new([1.0, 1.0], -1.5);
    let net = and.then(ActivationLayer::<1, _>::new(Relu));
    assert_eq!(net.forward(&[1.0, 1.0]), [1.0]);
    assert_eq!(net.forward(&[1.0, 0.0]), [0.0]);
}

#[test]
fn perceptron_works_with_f64() {
    let mut p = Perceptron::<2, f64>::new([0.0, 0.0], 0.0);
    let data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 0.0),
        ([1.0, 0.0], 0.0),
        ([1.0, 1.0], 1.0),
    ];
    assert!(p.fit(&data, 100, 0.1).is_some());
    for (input, target) in &data {
        assert_eq!(p.predict(input), *target);
    }
}
