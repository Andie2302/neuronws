use neuronws_layers::activation::{Identity, Relu};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::traits::ForwardPath;

#[test]
fn relu_layer_clips_each_element() {
    let layer = ActivationLayer::<3, _>::new(Relu);
    assert_eq!(layer.forward(&[-1.0, 0.0, 2.0]), [0.0, 0.0, 2.0]);
}

#[test]
fn identity_layer_changes_nothing() {
    let layer = ActivationLayer::<2, _>::new(Identity);
    assert_eq!(layer.forward(&[-1.5, 4.0]), [-1.5, 4.0]);
}
