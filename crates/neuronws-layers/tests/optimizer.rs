use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Momentum, Optimizer, Sgd};
use neuronws_layers::traits::{BackwardPathMut, ForwardPathMut, Trainable};

fn assert_close(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-6, "{a} gegen {b}");
}

fn one_gradient_step(layer: &mut DenseLayer<1, 1>) {
    layer.forward_mut(&[2.0]);
    layer.backward_mut(&[1.0]); // grad_w = 2, grad_b = 1
}

/// Ein Lernschritt für beliebige Optimierer: genau dafür ist das Trait da.
fn train_line<O: Optimizer<f32>>(optimizer: &mut O, epochs: usize) -> DenseLayer<1, 1> {
    let mut layer = DenseLayer::new([[0.0]], [0.0]);
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x: f32| ([x], [2.0 * x + 1.0]));
    for _ in 0..epochs {
        for (x, target) in &data {
            let y = *layer.forward_mut(x);
            layer.backward_mut(&MeanSquaredError.gradient(&y, target));
        }
        optimizer.step(&mut layer, data.len());
    }
    layer
}

fn epochs_until_close<O: Optimizer<f32>>(mut optimizer: O) -> Option<usize> {
    let mut layer = DenseLayer::<1, 1, f32>::new([[0.0]], [0.0]);
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x: f32| ([x], [2.0 * x + 1.0]));
    for epoch in 0..5000 {
        let [[w]] = *layer.weights();
        let [b] = *layer.bias();
        if (w - 2.0).abs() < 1e-3 && (b - 1.0).abs() < 1e-3 {
            return Some(epoch);
        }
        for (x, target) in &data {
            let y = *layer.forward_mut(x);
            layer.backward_mut(&MeanSquaredError.gradient(&y, target));
        }
        optimizer.step(&mut layer, data.len());
    }
    None
}

#[test]
fn sgd_and_momentum_both_work_through_the_trait() {
    for layer in [
        train_line(&mut Sgd::new(0.05), 300),
        train_line(&mut Momentum::<2>::new(0.02, 0.9), 300),
    ] {
        assert_close(layer.weights()[0][0], 2.0);
        assert_close(layer.bias()[0], 1.0);
    }
}

#[test]
fn momentum_step_follows_the_formula_and_remembers_velocity() {
    let mut layer = DenseLayer::new([[1.0]], [0.0]);
    let mut momentum = Momentum::<2>::new(0.1, 0.5);

    // Schritt 1: v = −η g, also v_w = −0.2 und v_b = −0.1
    one_gradient_step(&mut layer);
    momentum.step(&mut layer, 1);
    assert_close(layer.weights()[0][0], 1.0 - 0.2);
    assert_close(layer.bias()[0], -0.1);

    // Schritt 2: v = μ v − η g, also v_w = 0.5·(−0.2) − 0.2 = −0.3 und v_b = −0.15
    one_gradient_step(&mut layer);
    momentum.step(&mut layer, 1);
    assert_close(layer.weights()[0][0], 0.8 - 0.3);
    assert_close(layer.bias()[0], -0.1 - 0.15);
}

#[test]
fn momentum_clears_the_gradients_like_sgd() {
    let mut layer = DenseLayer::new([[1.0]], [0.0]);
    one_gradient_step(&mut layer);
    Momentum::<2>::new(0.1, 0.9).step(&mut layer, 1);
    assert_eq!(layer.grad_weights(), &[[0.0]]);
    assert_eq!(layer.grad_bias(), &[0.0]);
}

#[test]
fn momentum_with_zero_mu_is_plain_sgd() {
    let mut a = DenseLayer::new([[1.0]], [0.5]);
    let mut b = DenseLayer::new([[1.0]], [0.5]);
    one_gradient_step(&mut a);
    one_gradient_step(&mut b);

    Sgd::new(0.1).step(&mut a, 1);
    Momentum::<2>::new(0.1, 0.0).step(&mut b, 1);

    assert_eq!(a.weights(), b.weights());
    assert_eq!(a.bias(), b.bias());
}

#[test]
fn momentum_converges_in_fewer_rounds_than_sgd() {
    let sgd = epochs_until_close(Sgd::new(0.02)).expect("Sgd konvergiert");
    let momentum = epochs_until_close(Momentum::<2>::new(0.02, 0.9)).expect("Momentum konvergiert");
    assert!(
        momentum < sgd,
        "Momentum {momentum} Runden, Sgd {sgd} Runden"
    );
}

#[test]
#[should_panic(expected = "mehr als 1 Parameter")]
fn momentum_panics_when_p_is_too_small() {
    let mut layer = DenseLayer::new([[1.0]], [0.0]);
    Momentum::<1>::new(0.1, 0.9).step(&mut layer, 1);
}

#[test]
#[should_panic(expected = "hat 2 Parameter")]
fn momentum_panics_when_p_is_too_large() {
    let mut layer = DenseLayer::new([[1.0]], [0.0]);
    Momentum::<3>::new(0.1, 0.9).step(&mut layer, 1);
}

#[test]
fn trainable_still_reports_parameters_in_a_stable_order() {
    let mut layer = DenseLayer::new([[1.0, 2.0]], [3.0]);
    let mut seen = [0.0_f32; 3];
    let mut i = 0;
    layer.visit_parameters(&mut |p, _| {
        seen[i] = *p;
        i += 1;
    });
    assert_eq!(seen, [1.0, 2.0, 3.0]); // erst die Gewichte, dann der Bias
}
