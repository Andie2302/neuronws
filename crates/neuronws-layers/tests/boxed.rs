//! Geboxte Schichten verhalten sich genau wie ungeboxte. Braucht das Feature `alloc`.
#![cfg(feature = "alloc")]

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::optimizer::{HasLearningRate, Momentum, Sgd};
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{ForwardPath, Trainable};

type Plain = neuronws_layers::chain::Chain<
    neuronws_layers::chain::Chain<DenseLayer<1, 4>, ActivationLayer<4, Relu>>,
    DenseLayer<4, 1>,
>;
type Boxed = neuronws_layers::chain::Chain<
    neuronws_layers::chain::Chain<Box<DenseLayer<1, 4>>, Box<ActivationLayer<4, Relu>>>,
    Box<DenseLayer<4, 1>>,
>;

fn first() -> DenseLayer<1, 4> {
    DenseLayer::new([[0.5], [-0.3], [0.8], [0.1]], [0.1, 0.0, -0.1, 0.2])
}

fn last() -> DenseLayer<4, 1> {
    DenseLayer::new([[0.4, -0.6, 0.2, 0.9]], [0.0])
}

fn plain() -> Plain {
    first()
        .then(ActivationLayer::<4, _>::new(Relu))
        .then(last())
}

fn boxed() -> Boxed {
    Box::new(first())
        .then(Box::new(ActivationLayer::<4, _>::new(Relu)))
        .then(Box::new(last()))
}

fn data() -> [([f32; 1], [f32; 1]); 5] {
    [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x| ([x], [0.5 * x + 1.0]))
}

#[test]
fn a_boxed_net_calculates_the_same_forward_pass() {
    assert_eq!(plain().forward(&[0.7]), boxed().forward(&[0.7]));
}

#[test]
fn a_boxed_net_has_the_same_parameters() {
    assert_eq!(
        <Boxed as Trainable>::PARAMETER_COUNT,
        <Plain as Trainable>::PARAMETER_COUNT
    );
    assert_eq!(boxed().parameters(), plain().parameters());
}

#[test]
fn a_boxed_net_learns_exactly_like_an_unboxed_one() {
    let (mut plain, mut boxed) = (plain(), boxed());
    let mut a = Trainer::new(MeanSquaredError, Sgd::new(0.02), 5);
    let mut b = Trainer::new(MeanSquaredError, Sgd::new(0.02), 5);
    let data = data();
    let loss_plain = a.fit(&mut plain, &data, 50);
    let loss_boxed = b.fit(&mut boxed, &data, 50);

    assert_eq!(loss_plain, loss_boxed);
    assert_eq!(plain.parameters(), boxed.parameters());
}

#[test]
fn a_boxed_optimizer_works_too() {
    let mut net = plain();
    let momentum = Box::new(Momentum::for_model(&net, 0.01, 0.9));
    let mut trainer = Trainer::new(MeanSquaredError, momentum, 5);
    let data = data();
    let before = trainer.evaluate(&net, &data);
    let after = trainer.fit(&mut net, &data, 100);
    assert!(after < before);

    // Die Lernrate ist durch die Box hindurch erreichbar.
    trainer.optimizer_mut().set_learning_rate(0.5);
    assert_eq!(trainer.optimizer().learning_rate(), 0.5);
}

#[test]
fn a_chain_of_boxes_is_only_as_big_as_its_pointers() {
    // Das ist der Zweck der Boxen: Die Kette liegt auf dem Stack, die Schichten auf dem Heap.
    let pointer = core::mem::size_of::<usize>();
    assert_eq!(core::mem::size_of::<Boxed>(), 3 * pointer);
    // Dieselben Schichten ohne Box tragen alle Arrays in der Kette selbst.
    assert!(core::mem::size_of::<Plain>() > 5 * core::mem::size_of::<Boxed>());
}
