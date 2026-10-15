//! Der Gradiententest funktioniert für jede Operation, auch für selbst geschriebene.

mod common;

use common::{BrokenDouble, ScaleShift};
use neuronws_layers::activation::{Identity, Relu};
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::dense_layer::Linear;
use neuronws_layers::gradcheck::assert_gradients_match;
use neuronws_layers::standardize::Standardization;

#[test]
fn linear_f32() {
    let params = ([[0.5_f32, -1.0, 0.25], [2.0, 0.3, -0.7]], [0.1, -0.2]);
    assert_gradients_match(&Linear::<3, 2>::new(), params, [1.5, -0.5, 0.8], 1e-2, 2e-3);
}

#[test]
fn linear_f64() {
    let params = ([[0.5_f64, -1.0, 0.25], [2.0, 0.3, -0.7]], [0.1, -0.2]);
    assert_gradients_match(
        &Linear::<3, 2, f64>::new(),
        params,
        [1.5, -0.5, 0.8],
        1e-6,
        1e-7,
    );
}

#[test]
fn elementwise_identity_and_relu_away_from_the_kink() {
    let x = [-1.5_f32, 0.7, 2.0, -0.3];
    assert_gradients_match(
        &Elementwise::<[f32; 4], _>::new(Identity),
        (),
        x,
        1e-2,
        2e-3,
    );
    assert_gradients_match(&Elementwise::<[f32; 4], _>::new(Relu), (), x, 1e-2, 2e-3);
}

#[test]
fn scale_shift_f32_and_f64() {
    let op = ScaleShift::<3>::new();
    assert_gradients_match(
        &op,
        ([0.5_f32, -1.0, 2.0], [0.1, 0.2, 0.3]),
        [1.0, -2.0, 0.5],
        1e-2,
        2e-3,
    );
    let op = ScaleShift::<3, f64>::new();
    assert_gradients_match(
        &op,
        ([0.5_f64, -1.0, 2.0], [0.1, 0.2, 0.3]),
        [1.0, -2.0, 0.5],
        1e-6,
        1e-7,
    );
}

#[test]
fn standardization() {
    let block =
        neuronws_layers::standardize::Standardize::from_statistics([1.0_f32, -2.0], [2.0, 0.5]);
    let op: &Standardization<2> = block.operation();
    assert_gradients_match(op, (), [3.0, 1.0], 1e-2, 2e-3);
}

#[test]
#[should_panic(expected = "Ableitung nach Eingabe")]
fn a_wrong_backward_is_caught() {
    assert_gradients_match(&BrokenDouble, (), [1.0, 2.0], 1e-2, 1e-3);
}

#[cfg(any(feature = "std", feature = "libm"))]
mod smooth {
    use neuronws_layers::activation::{Sigmoid, Tanh};
    use neuronws_layers::softmax_layer::Softmax;

    use super::*;

    #[test]
    fn sigmoid_tanh_and_softmax() {
        let x = [0.4_f32, -1.2, 2.0, 0.1];
        assert_gradients_match(&Elementwise::<[f32; 4], _>::new(Sigmoid), (), x, 1e-2, 2e-3);
        assert_gradients_match(&Elementwise::<[f32; 4], _>::new(Tanh), (), x, 1e-2, 2e-3);
        assert_gradients_match(&Softmax::<4>::new(), (), x, 1e-2, 2e-3);
    }

    #[test]
    fn softmax_f64() {
        assert_gradients_match(&Softmax::<3, f64>::new(), (), [0.4, -1.2, 2.0], 1e-6, 1e-7);
    }
}
