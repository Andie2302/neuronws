use neuronws_layers::activation::{Activation, Identity, Relu, Step};

#[test]
fn identity_passes_values_through() {
    assert_eq!(Identity.apply(-2.5), -2.5);
    assert_eq!(Identity.derivative(-2.5), 1.0);
}

#[test]
fn step_switches_at_zero() {
    assert_eq!(Step.apply(-0.1), 0.0);
    assert_eq!(Step.apply(0.0), 1.0);
    assert_eq!(Step.apply(3.0), 1.0);
}

#[test]
fn step_has_no_gradient() {
    assert_eq!(Step.derivative(-1.0), 0.0);
    assert_eq!(Step.derivative(1.0), 0.0);
}

#[test]
fn relu_clips_negative_values() {
    assert_eq!(Relu.apply(-2.0), 0.0);
    assert_eq!(Relu.apply(2.0), 2.0);
    assert_eq!(Relu.derivative(-2.0), 0.0);
    assert_eq!(Relu.derivative(2.0), 1.0);
}
