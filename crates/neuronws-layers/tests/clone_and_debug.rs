use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Momentum, Optimizer, Sgd};
use neuronws_layers::random::Rng;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

fn learn_one_step(layer: &mut DenseLayer<1, 1, f32>) {
    let y = *layer.forward_mut(&[2.0]);
    layer.backward_mut(&MeanSquaredError.gradient(&y, &[10.0]));
    Sgd::new(0.1).step(layer, 1);
}

#[test]
fn a_clone_is_an_independent_snapshot() {
    let mut layer = DenseLayer::<1, 1, f32>::new([[1.0]], [0.0]);
    let snapshot = layer.clone();

    learn_one_step(&mut layer);

    assert_ne!(layer.weights(), snapshot.weights());
    assert_eq!(snapshot.weights(), &[[1.0]]);
    assert_eq!(snapshot.forward(&[2.0]), [2.0]);
}

#[test]
fn a_cloned_chain_keeps_working_and_stays_independent() {
    let mut net = DenseLayer::<1, 1, f32>::new([[1.0]], [0.0])
        .then(ActivationLayer::<1, _>::new(Relu))
        .then(DenseLayer::<1, 1, f32>::new([[1.0]], [0.0]));
    let copy = net.clone();

    for _ in 0..5 {
        let y = *net.forward_mut(&[1.0]);
        net.backward_mut(&MeanSquaredError.gradient(&y, &[5.0]));
        Sgd::new(0.1).step(&mut net, 1);
    }

    assert_eq!(copy.forward(&[1.0]), [1.0]);
    assert_ne!(net.forward(&[1.0]), [1.0]);
}

#[test]
fn a_cloned_optimizer_carries_its_velocity() {
    let mut layer = DenseLayer::<1, 1, f32>::new([[1.0]], [0.0]);
    let mut momentum = Momentum::<2>::new(0.1, 0.9);
    layer.forward_mut(&[2.0]);
    layer.backward_mut(&[1.0]);
    momentum.step(&mut layer, 1);

    // Zwei Kopien mit demselben Zustand verhalten sich gleich weiter.
    let (mut a, mut b) = (layer.clone(), layer.clone());
    let (mut ma, mut mb) = (momentum.clone(), momentum);
    for (layer, momentum) in [(&mut a, &mut ma), (&mut b, &mut mb)] {
        layer.forward_mut(&[2.0]);
        layer.backward_mut(&[1.0]);
        momentum.step(layer, 1);
    }
    assert_eq!(a.weights(), b.weights());
}

#[test]
fn a_cloned_rng_continues_with_the_same_sequence() {
    let mut rng = Rng::new(5);
    rng.next_u32();
    let mut copy = rng.clone();
    for _ in 0..10 {
        assert_eq!(rng.next_u32(), copy.next_u32());
    }
}

#[test]
fn debug_output_names_the_type_and_shows_its_values() {
    let layer = DenseLayer::<1, 1, f32>::new([[1.5]], [0.25]);
    let text = format!("{layer:?}");
    assert!(text.starts_with("DenseLayer"), "{text}");
    assert!(text.contains("1.5") && text.contains("0.25"), "{text}");

    assert!(format!("{:?}", Sgd::new(0.1_f32)).contains("learning_rate"));
    assert!(format!("{:?}", Rng::new(1)).starts_with("Rng"));
}
