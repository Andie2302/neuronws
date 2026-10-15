//! Sigmoid und Tanh. Sie brauchen `exp`/`tanh`, also das Feature `std` oder `libm`.
//!
//! Start: `cargo test -p neuronws-layers --features libm`
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::{Activation, Sigmoid, Tanh};

const EPS: f32 = 1e-3;

fn assert_close(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-4, "{a} gegen {b}");
}

fn check_derivative(activation: &impl Activation) {
    for z in [-3.0, -1.0, -0.2, 0.0, 0.4, 1.5, 4.0] {
        let numeric = (activation.apply(z + EPS) - activation.apply(z - EPS)) / (2.0 * EPS);
        assert_close(activation.derivative(z), numeric);
    }
}

#[test]
fn sigmoid_is_one_half_at_zero() {
    assert_close(Sigmoid.apply(0.0), 0.5);
    assert_close(Sigmoid.derivative(0.0), 0.25);
}

#[test]
fn sigmoid_is_symmetric_around_one_half() {
    for z in [0.5, 1.0, 3.0, 8.0] {
        assert_close(Sigmoid.apply(z) + Sigmoid.apply(-z), 1.0);
    }
}

#[test]
fn sigmoid_saturates_without_producing_nan() {
    assert_eq!(Sigmoid.apply(1000.0), 1.0);
    assert_eq!(Sigmoid.apply(-1000.0), 0.0);
    assert_eq!(Sigmoid.derivative(1000.0), 0.0);
    assert_eq!(Sigmoid.derivative(-1000.0), 0.0);
}

#[test]
fn sigmoid_derivative_matches_numeric_derivative() {
    check_derivative(&Sigmoid);
}

#[test]
fn tanh_is_zero_at_zero_with_slope_one() {
    assert_close(Tanh.apply(0.0), 0.0);
    assert_close(Tanh.derivative(0.0), 1.0);
}

#[test]
fn tanh_is_odd_and_bounded() {
    for z in [0.5, 1.0, 3.0] {
        assert_close(Tanh.apply(z), -Tanh.apply(-z));
    }
    assert_eq!(Tanh.apply(1000.0), 1.0);
    assert_eq!(Tanh.apply(-1000.0), -1.0);
}

#[test]
fn tanh_derivative_matches_numeric_derivative() {
    check_derivative(&Tanh);
}
