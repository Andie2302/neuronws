//! `Block` macht aus jeder `Operation` eine vollständige, trainierbare Schicht.

mod common;

use common::ScaleShift;
use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::init::Zeros;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::random::Rng;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

type Scale3 = Block<ScaleShift<3>>;

fn scale3() -> Scale3 {
    Block::initialized(ScaleShift::new(), &Zeros, &mut Rng::new(0))
}

#[test]
fn a_custom_operation_becomes_a_working_layer() {
    let layer = Block::from_parts(ScaleShift::<2>::new(), ([2.0_f32, 3.0], [10.0, 20.0]));
    assert_eq!(layer.forward(&[1.0, 1.0]), [12.0, 23.0]);
}

#[test]
fn initialization_is_up_to_the_operation() {
    let layer = scale3();
    assert_eq!(layer.params(), &([1.0; 3], [0.0; 3]));
    assert_eq!(layer.forward(&[4.0, 5.0, 6.0]), [4.0, 5.0, 6.0]);
}

#[test]
fn parameters_are_counted_and_visited_in_order() {
    assert_eq!(<Scale3 as Trainable>::PARAMETER_COUNT, 6);
    assert_eq!(<ActivationLayer<4, Relu> as Trainable>::PARAMETER_COUNT, 0);
    assert_eq!(<DenseLayer<2, 3> as Trainable>::PARAMETER_COUNT, 9);

    let mut layer = Block::from_parts(ScaleShift::<2>::new(), ([1.0_f32, 2.0], [3.0, 4.0]));
    let mut seen = vec![];
    layer.visit_parameters(&mut |p, _| seen.push(*p));
    assert_eq!(seen, [1.0, 2.0, 3.0, 4.0]); // erst a, dann b
}

#[test]
fn gradients_are_collected_and_cleared() {
    let mut layer = Block::from_parts(ScaleShift::<2>::new(), ([2.0_f32, 3.0], [0.0, 0.0]));
    layer.forward_mut(&[5.0, 7.0]);
    let grad_input = *layer.backward_mut(&[1.0, 10.0]);

    assert_eq!(grad_input, [2.0, 30.0]); // g · a
    assert_eq!(layer.grads(), &([5.0, 70.0], [1.0, 10.0])); // (g · x, g)

    layer.forward_mut(&[5.0, 7.0]);
    layer.backward_mut(&[1.0, 10.0]);
    assert_eq!(layer.grads(), &([10.0, 140.0], [2.0, 20.0])); // addiert, nicht überschrieben

    layer.zero_grad();
    assert_eq!(layer.grads(), &([0.0; 2], [0.0; 2]));
}

#[test]
fn a_custom_layer_alone_learns_a_line() {
    // y = 3x + 2 mit nur einer ScaleShift<1>-Schicht: lernt a → 3 und b → 2
    let mut layer = Block::from_parts(ScaleShift::<1>::new(), ([1.0_f32], [0.0]));
    let mut sgd = Sgd::new(0.05);
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x: f32| ([x], [3.0 * x + 2.0]));

    for _ in 0..400 {
        for (x, target) in &data {
            let y = *layer.forward_mut(x);
            layer.backward_mut(&MeanSquaredError.gradient(&y, target));
        }
        sgd.step(&mut layer, data.len());
    }

    let ([a], [b]) = *layer.params();
    assert!(
        (a - 3.0).abs() < 1e-3 && (b - 2.0).abs() < 1e-3,
        "a = {a}, b = {b}"
    );
}

#[test]
fn a_custom_layer_chains_with_the_built_in_ones() {
    let mut net = DenseLayer::<2, 3>::new([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0; 3])
        .then(scale3())
        .then(ActivationLayer::<3, _>::new(Relu));
    assert_eq!(net.forward(&[2.0, -3.0]), [2.0, 0.0, 0.0]);

    // ... und über die ganze Kette lässt sich trainieren
    let mut sgd = Sgd::new(0.05);
    let y = *net.forward_mut(&[2.0, 1.0]);
    net.backward_mut(&MeanSquaredError.gradient(&y, &[0.0, 0.0, 0.0]));
    sgd.step(&mut net, 1);
    assert_ne!(net.forward(&[2.0, 1.0]), y);
    assert_eq!(<Chain3 as Trainable>::PARAMETER_COUNT, 9 + 6);
}

type Chain3 = neuronws_layers::chain::Chain<
    neuronws_layers::chain::Chain<DenseLayer<2, 3>, Scale3>,
    ActivationLayer<3, Relu>,
>;

#[test]
fn blocks_can_be_cloned_and_printed() {
    let mut layer = scale3();
    let snapshot = layer.clone();
    layer.forward_mut(&[1.0, 2.0, 3.0]);
    layer.backward_mut(&[1.0, 1.0, 1.0]);
    assert_ne!(layer.grads(), snapshot.grads());
    assert!(format!("{snapshot:?}").starts_with("Block"));
}

#[test]
fn the_operation_can_be_read_back() {
    let layer = scale3();
    let _: &ScaleShift<3> = layer.operation();
}

#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn an_operation_without_parameters_has_none_to_visit() {
    use neuronws_layers::softmax_layer::SoftmaxLayer;

    assert_eq!(<SoftmaxLayer<4> as Trainable>::PARAMETER_COUNT, 0);
    let mut layer = SoftmaxLayer::<4>::new();
    let mut visited = 0;
    layer.visit_parameters(&mut |_, _| visited += 1);
    assert_eq!(visited, 0);
}
