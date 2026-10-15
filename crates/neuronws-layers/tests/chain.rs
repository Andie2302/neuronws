use neuronws_layers::activation::{Relu, Step};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::{Chain, LayerExt};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::traits::ForwardPath;

#[test]
fn chain_feeds_output_of_first_into_second() {
    // 2 -> 2 -> ReLU
    let dense = DenseLayer::new([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0]);
    let relu = ActivationLayer::<2, _>::new(Relu);
    let net = Chain::new(dense, relu);

    assert_eq!(net.forward(&[3.0, 5.0]), [3.0, 0.0]);
}

#[test]
fn dense_followed_by_step_is_a_perceptron() {
    // AND-Gatter: 1 nur, wenn beide Eingänge 1 sind
    let and = Chain::new(
        DenseLayer::new([[1.0, 1.0]], [-1.5]),
        ActivationLayer::<1, _>::new(Step),
    );

    assert_eq!(and.forward(&[0.0, 0.0]), [0.0]);
    assert_eq!(and.forward(&[1.0, 0.0]), [0.0]);
    assert_eq!(and.forward(&[0.0, 1.0]), [0.0]);
    assert_eq!(and.forward(&[1.0, 1.0]), [1.0]);
}

#[test]
fn chains_can_be_nested() {
    // Eine Chain ist selbst wieder ein Layer mit ForwardPath
    let double = DenseLayer::new([[2.0]], [0.0]);
    let plus_one = DenseLayer::new([[1.0]], [1.0]);
    let negate = DenseLayer::new([[-1.0]], [0.0]);

    let net = Chain::new(Chain::new(double, plus_one), negate);
    assert_eq!(net.forward(&[3.0]), [-7.0]);
}

#[test]
fn then_is_shorthand_for_chain_new() {
    let make_dense = || DenseLayer::new([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0]);
    let make_relu = || ActivationLayer::<2, _>::new(Relu);

    let with_then = make_dense().then(make_relu());
    let with_new = Chain::new(make_dense(), make_relu());

    assert_eq!(
        with_then.forward(&[3.0, 5.0]),
        with_new.forward(&[3.0, 5.0])
    );
}

#[test]
fn then_can_be_chained_repeatedly() {
    let net = DenseLayer::new([[2.0]], [0.0])
        .then(DenseLayer::new([[1.0]], [1.0]))
        .then(DenseLayer::new([[-1.0]], [0.0]));

    assert_eq!(net.forward(&[3.0]), [-7.0]);
}
