//! `Trainable`: Parameter besuchen, lesen, festhalten und zurückspielen.

mod common;

use common::ScaleShift;
use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::block::Block;
use neuronws_layers::chain::{Chain, LayerExt};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};
use neuronws_layers::values::Values;

type Net = Chain<DenseLayer<2, 3>, Chain<ActivationLayer<3, Relu>, DenseLayer<3, 1>>>;

fn net() -> Net {
    DenseLayer::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [0.1, 0.2, 0.3])
        .then(ActivationLayer::<3, _>::new(Relu).then(DenseLayer::new([[7.0, 8.0, 9.0]], [0.4])))
}

#[test]
fn the_parameter_count_always_follows_from_the_shape() {
    fn shape_len<M: Trainable>() -> usize {
        <M::Shape as Values<M::Scalar>>::LEN
    }
    assert_eq!(<Net as Trainable>::PARAMETER_COUNT, shape_len::<Net>());
    assert_eq!(<Net as Trainable>::PARAMETER_COUNT, 9 + 4);
    assert_eq!(
        <Block<ScaleShift<5>> as Trainable>::PARAMETER_COUNT,
        shape_len::<Block<ScaleShift<5>>>()
    );
}

#[test]
fn parameters_come_in_the_order_of_the_visit() {
    let mut net = net();
    let mut visited = vec![];
    net.visit_parameters(&mut |p, _| visited.push(*p));

    let shape = net.parameters();
    let flat: Vec<f32> = (0..<Net as Trainable>::PARAMETER_COUNT)
        .map(|i| shape.value(i))
        .collect();
    assert_eq!(flat, visited);
    assert_eq!(
        visited,
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3, 7.0, 8.0, 9.0, 0.4
        ]
    );
}

#[test]
fn parameters_can_be_saved_and_restored() {
    let mut net = net();
    let saved = net.parameters();
    let before = net.forward(&[1.0, -1.0]);

    // Das Netz lernt etwas dazu ...
    net.forward_mut(&[1.0, -1.0]);
    net.backward_mut(&[1.0]);
    net.visit_parameters(&mut |p, g| *p -= 0.5 * *g);
    assert_ne!(net.forward(&[1.0, -1.0]), before);

    // ... und kehrt zum gemerkten Stand zurück.
    net.set_parameters(&saved);
    assert_eq!(net.forward(&[1.0, -1.0]), before);
    assert_eq!(net.parameters(), saved);
}

#[test]
fn restoring_parameters_leaves_the_gradients_alone() {
    let mut net = net();
    net.forward_mut(&[1.0, 1.0]);
    net.backward_mut(&[1.0]);

    let mut grads_before = vec![];
    net.inspect_parameters(&mut |_, g| grads_before.push(g));
    assert!(grads_before.iter().any(|&g| g != 0.0));

    let zeros = <<Net as Trainable>::Shape as Values<f32>>::zeros();
    net.set_parameters(&zeros);

    let mut grads_after = vec![];
    let mut params_after = vec![];
    net.inspect_parameters(&mut |p, g| {
        params_after.push(p);
        grads_after.push(g);
    });
    assert_eq!(grads_before, grads_after);
    assert!(params_after.iter().all(|&p| p == 0.0));
}

#[test]
fn the_visitor_may_change_the_gradients() {
    let mut layer = DenseLayer::new([[1.0, 1.0]], [0.0]);
    layer.forward_mut(&[3.0, 4.0]);
    layer.backward_mut(&[2.0]);
    assert_eq!(layer.grad_weights(), &[[6.0, 8.0]]);

    layer.visit_parameters(&mut |_, gradient| *gradient *= 0.5);
    assert_eq!(layer.grad_weights(), &[[3.0, 4.0]]);
    assert_eq!(layer.grad_bias(), &[1.0]);
}

#[test]
fn inspect_sees_what_visit_sees() {
    let mut net = net();
    net.forward_mut(&[0.5, 2.0]);
    net.backward_mut(&[1.0]);

    let mut by_visit = vec![];
    net.visit_parameters(&mut |p, g| by_visit.push((*p, *g)));
    let mut by_inspect = vec![];
    net.inspect_parameters(&mut |p, g| by_inspect.push((p, g)));
    assert_eq!(by_visit, by_inspect);
}
