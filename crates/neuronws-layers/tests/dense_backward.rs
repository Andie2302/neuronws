//! Prüft den Backward-Pass der Dense-Schicht gegen numerische Gradienten.
//!
//! Idee: `∂L/∂p ≈ (L(p + ε) − L(p − ε)) / 2ε`. Das ist langsam, aber so einfach,
//! dass man es kaum falsch machen kann.

use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

const W: [[f32; 2]; 3] = [[0.5, -1.0], [2.0, 0.3], [-0.7, 1.1]];
const B: [f32; 3] = [0.1, -0.2, 0.3];
const X: [f32; 2] = [1.5, -0.5];
const TARGET: [f32; 3] = [1.0, 0.0, -1.0];
const EPS: f32 = 1e-2;
const TOLERANCE: f32 = 1e-3;

/// L = ½ Σ (y − t)²
fn loss(weights: [[f32; 2]; 3], bias: [f32; 3], x: [f32; 2]) -> f32 {
    let y = DenseLayer::new(weights, bias).forward(&x);
    0.5 * y
        .iter()
        .zip(TARGET)
        .map(|(y, t)| (y - t) * (y - t))
        .sum::<f32>()
}

fn numeric(f: impl Fn(f32) -> f32) -> f32 {
    (f(EPS) - f(-EPS)) / (2.0 * EPS)
}

fn assert_close(analytic: f32, numeric: f32, what: &str) {
    assert!(
        (analytic - numeric).abs() < TOLERANCE,
        "{what}: analytisch {analytic}, numerisch {numeric}"
    );
}

/// Ein Forward- und ein Backward-Durchlauf mit dem Gradienten ∂L/∂y = y − t.
fn run_backward(layer: &mut DenseLayer<2, 3>) -> [f32; 2] {
    let y = layer.forward_mut(&X);
    let grad_output = core::array::from_fn(|j| y[j] - TARGET[j]);
    *layer.backward_mut(&grad_output)
}

#[test]
fn weight_gradients_match_numeric_gradients() {
    let mut layer = DenseLayer::new(W, B);
    run_backward(&mut layer);

    for j in 0..3 {
        for i in 0..2 {
            let n = numeric(|d| {
                let mut w = W;
                w[j][i] += d;
                loss(w, B, X)
            });
            assert_close(layer.grad_weights()[j][i], n, "grad_weights");
        }
    }
}

#[test]
fn bias_gradients_match_numeric_gradients() {
    let mut layer = DenseLayer::new(W, B);
    run_backward(&mut layer);

    for j in 0..3 {
        let n = numeric(|d| {
            let mut b = B;
            b[j] += d;
            loss(W, b, X)
        });
        assert_close(layer.grad_bias()[j], n, "grad_bias");
    }
}

#[test]
fn input_gradients_match_numeric_gradients() {
    let mut layer = DenseLayer::new(W, B);
    let grad_input = run_backward(&mut layer);

    for i in 0..2 {
        let n = numeric(|d| {
            let mut x = X;
            x[i] += d;
            loss(W, B, x)
        });
        assert_close(grad_input[i], n, "grad_input");
    }
}

#[test]
fn gradients_accumulate_until_zero_grad() {
    let mut layer = DenseLayer::new(W, B);
    run_backward(&mut layer);
    let once = *layer.grad_bias();

    run_backward(&mut layer);
    for (twice, once) in layer.grad_bias().iter().zip(once) {
        assert_close(*twice, 2.0 * once, "accumulated grad_bias");
    }

    layer.zero_grad();
    assert_eq!(layer.grad_bias(), &[0.0; 3]);
    assert_eq!(layer.grad_weights(), &[[0.0; 2]; 3]);
}
