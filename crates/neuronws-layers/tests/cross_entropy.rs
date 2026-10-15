//! Binäre Kreuzentropie. Braucht `ln`, also das Feature `std` oder `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::Sigmoid;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{BinaryCrossEntropy, Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

fn assert_close(a: f32, b: f32, tolerance: f32) {
    assert!((a - b).abs() < tolerance, "{a} gegen {b}");
}

#[test]
fn value_is_minus_log_of_the_probability_given_to_the_right_class() {
    // y = 0.5 für beide Klassen: L = ln 2
    assert_close(
        BinaryCrossEntropy.value(&[0.5], &[1.0]),
        core::f32::consts::LN_2,
        1e-6,
    );
    assert_close(
        BinaryCrossEntropy.value(&[0.5], &[0.0]),
        core::f32::consts::LN_2,
        1e-6,
    );
    // y = 0.9 und Ziel 1: L = −ln 0.9
    assert_close(
        BinaryCrossEntropy.value(&[0.9], &[1.0]),
        -0.9_f32.ln(),
        1e-6,
    );
}

#[test]
fn confident_wrong_answers_cost_far_more_than_confident_right_ones() {
    let right = BinaryCrossEntropy.value(&[0.99], &[1.0]);
    let wrong = BinaryCrossEntropy.value(&[0.01], &[1.0]);
    assert!(wrong > 100.0 * right, "{wrong} gegen {right}");
}

#[test]
fn gradient_matches_numeric_gradient() {
    const Y: [f32; 3] = [0.3, 0.8, 0.55];
    const T: [f32; 3] = [1.0, 1.0, 0.0];
    const EPS: f32 = 1e-3;

    let analytic = BinaryCrossEntropy.gradient(&Y, &T);
    for (i, a) in analytic.iter().enumerate() {
        let mut up = Y;
        up[i] += EPS;
        let mut down = Y;
        down[i] -= EPS;
        let numeric =
            (BinaryCrossEntropy.value(&up, &T) - BinaryCrossEntropy.value(&down, &T)) / (2.0 * EPS);
        assert_close(*a, numeric, 5e-3);
    }
}

#[test]
fn saturated_outputs_do_not_produce_nan_or_infinity() {
    for y in [0.0_f32, 1.0] {
        for t in [0.0_f32, 1.0] {
            assert!(
                BinaryCrossEntropy.value(&[y], &[t]).is_finite(),
                "y={y} t={t}"
            );
            assert!(
                BinaryCrossEntropy.gradient(&[y], &[t])[0].is_finite(),
                "y={y} t={t}"
            );
        }
    }
}

#[test]
fn works_with_f64() {
    let loss = BinaryCrossEntropy.value(&[0.5_f64], &[1.0]);
    assert!((loss - core::f64::consts::LN_2).abs() < 1e-12);
}

fn epochs_until_correct(loss: &impl Loss) -> Option<usize> {
    let data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];
    let mut net = DenseLayer::new(
        [[0.5, -0.4], [-0.3, 0.6], [0.8, 0.2], [-0.6, -0.7]],
        [0.1, 0.1, -0.1, 0.2],
    )
    .then(ActivationLayer::<4, _>::new(Sigmoid))
    .then(DenseLayer::new([[0.5, -0.5, 0.3, -0.2]], [0.0]))
    .then(ActivationLayer::<1, _>::new(Sigmoid));
    let mut sgd = Sgd::new(0.3);

    for epoch in 0..20_000 {
        if data
            .iter()
            .all(|(x, [t])| (net.forward(x)[0] - t).abs() < 0.4)
        {
            return Some(epoch);
        }
        for (x, target) in &data {
            let y = *net.forward_mut(x);
            net.backward_mut(&loss.gradient(&y, target));
        }
        sgd.step(&mut net, data.len());
    }
    None
}

#[test]
fn cross_entropy_learns_xor_faster_than_mse_with_sigmoid() {
    let bce = epochs_until_correct(&BinaryCrossEntropy).expect("BCE lernt XOR");
    let mse = epochs_until_correct(&MeanSquaredError).expect("MSE lernt XOR");
    assert!(bce < mse, "BCE {bce} Runden, MSE {mse} Runden");
}
