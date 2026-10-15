//! Dieselben Schichten rechnen auch mit `f64`.

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::scalar::Scalar;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

#[test]
fn default_scalar_type_is_f32() {
    let layer: DenseLayer<1, 1> = DenseLayer::new([[2.0]], [0.0]);
    let y: [f32; 1] = layer.forward(&[3.0]);
    assert_eq!(y, [6.0]);
}

#[test]
fn dense_layer_computes_in_f64() {
    let layer = DenseLayer::<2, 1, f64>::new([[0.1, 0.2]], [0.3]);
    let y: [f64; 1] = layer.forward(&[1.0, 1.0]);
    assert_eq!(y, [0.1 + 0.2 + 0.3]);
}

#[test]
fn f64_keeps_precision_that_f32_cannot() {
    let big_f32 = DenseLayer::<1, 1, f32>::new([[1.0]], [0.0]).forward(&[16_777_216.0])[0] + 1.0;
    let big_f64 = DenseLayer::<1, 1, f64>::new([[1.0]], [0.0]).forward(&[16_777_216.0])[0] + 1.0;
    assert_eq!(big_f32, 16_777_216.0); // 2^24 + 1 passt nicht mehr in 24 Bit Mantisse
    assert_eq!(big_f64, 16_777_217.0);
}

#[test]
fn f64_chain_with_turbofish_and_inferred_scalar() {
    let net = DenseLayer::<2, 3, f64>::new([[1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]], [0.0; 3])
        .then(ActivationLayer::<3, _, f64>::new(Relu))
        .then(DenseLayer::<3, 1, f64>::new([[1.0, 1.0, 1.0]], [0.0]));
    assert_eq!(net.forward(&[3.0, 1.0]), [2.0 + 2.0 + 0.0]);
}

#[test]
fn f64_learns_a_line_to_high_precision() {
    let mut layer = DenseLayer::<1, 1, f64>::new([[0.0]], [0.0]);
    let mut sgd = Sgd::<f64>::new(0.05);
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x: f64| ([x], [2.0 * x + 1.0]));

    for _ in 0..500 {
        for (x, target) in &data {
            let y = *layer.forward_mut(x);
            let gradient = MeanSquaredError.gradient(&y, target);
            layer.backward_mut(&gradient);
        }
        sgd.step(&mut layer, data.len());
    }

    let [[w]] = *layer.weights();
    let [b] = *layer.bias();
    // So genau wird f32 nie: dessen Rundungsfehler liegt bei ~1e-7
    assert!((w - 2.0).abs() < 1e-9, "w = {w}");
    assert!((b - 1.0).abs() < 1e-9, "b = {b}");
}

#[test]
fn scalar_constants_and_conversions() {
    assert_eq!(<f32 as Scalar>::TWO, 2.0);
    assert_eq!(<f64 as Scalar>::from_usize(3), 3.0);
    assert_eq!(<f64 as Scalar>::from_f32(0.5), 0.5);
    assert_eq!(<f32 as Scalar>::to_f64(0.25), 0.25);
    assert_eq!(<f64 as Scalar>::to_f64(0.1), 0.1);
    // f32 -> f64 verändert den Wert nicht, auch wenn er nicht genau darstellbar ist
    assert_eq!(<f32 as Scalar>::to_f64(0.1_f32), f64::from(0.1_f32));
}

#[cfg(any(feature = "std", feature = "libm"))]
mod smooth {
    use neuronws_layers::activation::{Activation, Sigmoid, Tanh};
    use neuronws_layers::random::Rng;

    use super::*;

    #[test]
    fn sigmoid_and_tanh_derivatives_are_exact_in_f64() {
        const EPS: f64 = 1e-6;
        for z in [-2.0, -0.5, 0.0, 0.7, 3.0] {
            for (name, analytic, numeric) in [
                (
                    "sigmoid",
                    Sigmoid.derivative(z),
                    (Sigmoid.apply(z + EPS) - Sigmoid.apply(z - EPS)) / (2.0 * EPS),
                ),
                (
                    "tanh",
                    Tanh.derivative(z),
                    (Tanh.apply(z + EPS) - Tanh.apply(z - EPS)) / (2.0 * EPS),
                ),
            ] {
                assert!((analytic - numeric).abs() < 1e-8, "{name} bei {z}");
            }
        }
    }

    #[test]
    fn xavier_works_for_f64() {
        let layer = DenseLayer::<2, 4, f64>::xavier(&mut Rng::new(5));
        assert!(layer.weights().iter().flatten().all(|w| w.abs() <= 1.0));
        assert!(layer.weights().iter().flatten().any(|w| w.abs() > 0.1));
    }
}
