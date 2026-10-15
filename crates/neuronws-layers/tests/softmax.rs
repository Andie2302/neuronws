//! Softmax-Schicht und Kreuzentropie für mehrere Klassen. Braucht `exp`/`ln`.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::Tanh;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{CategoricalCrossEntropy, Loss};
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::softmax_layer::SoftmaxLayer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

fn assert_close(a: f32, b: f32, tolerance: f32) {
    assert!((a - b).abs() < tolerance, "{a} gegen {b}");
}

#[test]
fn outputs_are_probabilities_that_sum_to_one() {
    let y = SoftmaxLayer::<4>::new().forward(&[1.0, 2.0, 3.0, -1.0]);
    assert!(y.iter().all(|&p| p > 0.0 && p < 1.0));
    assert_close(y.iter().sum(), 1.0, 1e-6);
}

#[test]
fn larger_inputs_get_larger_probabilities() {
    let y = SoftmaxLayer::<3>::new().forward(&[0.5, 2.0, 1.0]);
    assert!(y[1] > y[2] && y[2] > y[0]);
}

#[test]
fn equal_inputs_give_a_uniform_distribution() {
    let y = SoftmaxLayer::<4>::new().forward(&[3.0; 4]);
    for p in y {
        assert_close(p, 0.25, 1e-6);
    }
}

#[test]
fn adding_a_constant_changes_nothing() {
    let layer = SoftmaxLayer::<3>::new();
    let a = layer.forward(&[1.0, 2.0, 3.0]);
    let b = layer.forward(&[11.0, 12.0, 13.0]);
    for (x, y) in a.iter().zip(b) {
        assert_close(*x, y, 1e-6);
    }
}

#[test]
fn huge_inputs_do_not_overflow() {
    let y = SoftmaxLayer::<3>::new().forward(&[1000.0, 1001.0, 1002.0]);
    assert!(y.iter().all(|p| p.is_finite()));
    assert_close(y.iter().sum(), 1.0, 1e-6);
    assert!(y[2] > y[1] && y[1] > y[0]);
}

#[test]
fn works_with_f64() {
    let y = SoftmaxLayer::<2, f64>::new().forward(&[0.0, 0.0]);
    assert_eq!(y, [0.5, 0.5]);
}

#[test]
fn backward_matches_numeric_gradient() {
    // Ein beliebiger Verlust darüber: L = Σ c[i] · y[i]
    const Z: [f32; 3] = [0.3, -1.2, 0.8];
    const C: [f32; 3] = [1.0, -2.0, 0.5];
    const EPS: f32 = 1e-3;
    let loss = |z: [f32; 3]| -> f32 {
        let y = SoftmaxLayer::<3>::new().forward(&z);
        y.iter().zip(C).map(|(y, c)| y * c).sum()
    };

    let mut layer = SoftmaxLayer::<3>::new();
    layer.forward_mut(&Z);
    let analytic = *layer.backward_mut(&C);

    for (i, a) in analytic.iter().enumerate() {
        let mut up = Z;
        up[i] += EPS;
        let mut down = Z;
        down[i] -= EPS;
        assert_close(*a, (loss(up) - loss(down)) / (2.0 * EPS), 1e-3);
    }
}

#[test]
fn cross_entropy_value_is_minus_log_of_the_right_class() {
    let loss = CategoricalCrossEntropy.value(&[0.2, 0.7, 0.1], &[0.0, 1.0, 0.0]);
    assert_close(loss, -0.7_f32.ln(), 1e-6);
}

#[test]
fn cross_entropy_gradient_matches_numeric_gradient() {
    const Y: [f32; 3] = [0.2, 0.5, 0.3];
    const T: [f32; 3] = [0.0, 1.0, 0.0];
    const EPS: f32 = 1e-3;
    let analytic = CategoricalCrossEntropy.gradient(&Y, &T);
    for (i, a) in analytic.iter().enumerate() {
        let mut up = Y;
        up[i] += EPS;
        let mut down = Y;
        down[i] -= EPS;
        let numeric = (CategoricalCrossEntropy.value(&up, &T)
            - CategoricalCrossEntropy.value(&down, &T))
            / (2.0 * EPS);
        assert_close(*a, numeric, 5e-3);
    }
}

#[test]
fn zero_probability_does_not_produce_infinity() {
    let target = [1.0_f32, 0.0];
    assert!(
        CategoricalCrossEntropy
            .value(&[0.0, 1.0], &target)
            .is_finite()
    );
    assert!(CategoricalCrossEntropy.gradient(&[0.0, 1.0], &target)[0].is_finite());
}

#[test]
fn softmax_with_cross_entropy_gives_the_famous_simple_gradient_y_minus_t() {
    // Die beiden Ableitungen kürzen sich zu ∂L/∂z = y − t.
    let target = [0.0, 1.0, 0.0];
    let mut layer = SoftmaxLayer::<3>::new();
    let y = *layer.forward_mut(&[0.4, -0.3, 1.1]);
    let g = CategoricalCrossEntropy.gradient(&y, &target);
    let grad_z = *layer.backward_mut(&g);
    for i in 0..3 {
        assert_close(grad_z[i], y[i] - target[i], 1e-5);
    }
}

#[test]
fn network_learns_to_classify_three_regions() {
    // Drei Punktwolken im Plan, eine pro Klasse: links unten, rechts unten, oben.
    let points: [([f32; 2], usize); 9] = [
        ([-1.0, -1.0], 0),
        ([-1.2, -0.8], 0),
        ([-0.8, -1.1], 0),
        ([1.0, -1.0], 1),
        ([1.2, -0.9], 1),
        ([0.9, -1.2], 1),
        ([0.0, 1.0], 2),
        ([0.1, 1.3], 2),
        ([-0.1, 0.9], 2),
    ];
    let one_hot = |class: usize| -> [f32; 3] { core::array::from_fn(|i| f32::from(i == class)) };

    let mut net = DenseLayer::<2, 4>::new(
        [[0.3, -0.2], [-0.4, 0.5], [0.2, 0.6], [-0.5, -0.3]],
        [0.0; 4],
    )
    .then(ActivationLayer::<4, _>::new(Tanh))
    .then(DenseLayer::<4, 3>::new(
        [
            [0.2, -0.3, 0.1, 0.4],
            [-0.1, 0.4, -0.2, 0.3],
            [0.3, 0.1, 0.2, -0.4],
        ],
        [0.0; 3],
    ))
    .then(SoftmaxLayer::<3>::new());
    let mut sgd = Sgd::new(0.5);

    for _ in 0..300 {
        for (x, class) in &points {
            let y = *net.forward_mut(x);
            net.backward_mut(&CategoricalCrossEntropy.gradient(&y, &one_hot(*class)));
        }
        sgd.step(&mut net, points.len());
    }

    for (x, class) in &points {
        let y = net.forward(x);
        let predicted = (0..3).max_by(|&a, &b| y[a].total_cmp(&y[b])).unwrap();
        assert_eq!(
            predicted, *class,
            "{x:?} wurde als {predicted} erkannt: {y:?}"
        );
        assert!(y[*class] > 0.8, "{x:?}: nur {} Sicherheit", y[*class]);
    }
}
