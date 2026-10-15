//! Weitere Fehlerfunktionen: Mean Absolute Error, Huber und Softmax-Kreuzentropie.

use neuronws_layers::gradcheck::assert_loss_gradient_matches;
use neuronws_layers::loss::{Huber, Loss, MeanAbsoluteError, MeanSquaredError};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} gegen {b}");
}

#[test]
fn mae_is_the_mean_of_the_absolute_differences() {
    let output = [1.0_f64, 5.0, -2.0, 0.0];
    let target = [2.0, 3.0, -2.0, 4.0];
    // |−1| + |2| + |0| + |−4| = 7, geteilt durch 4
    close(MeanAbsoluteError.value(&output, &target), 7.0 / 4.0);
    // Der Betrag des Gradienten ist überall 1/N, bei Gleichheit 0.
    assert_eq!(
        MeanAbsoluteError.gradient(&output, &target),
        [-0.25, 0.25, 0.0, -0.25]
    );
}

#[test]
fn mae_gradient_matches_the_numeric_one_away_from_the_kink() {
    assert_loss_gradient_matches(
        &MeanAbsoluteError,
        [1.0_f64, 5.0, -2.0],
        [2.0, 3.0, 0.5],
        1e-6,
        1e-7,
    );
}

#[test]
fn mae_treats_outliers_less_harshly_than_mse() {
    let target = [0.0_f64; 3];
    let with_outlier = [0.1, 0.1, 10.0];
    let mse = MeanSquaredError.value(&with_outlier, &target);
    let mae = MeanAbsoluteError.value(&with_outlier, &target);
    assert!(mse > 30.0 && mae < 4.0);
}

#[test]
fn huber_is_quadratic_close_and_linear_far_away() {
    let huber = Huber(1.0);
    let target = [0.0_f64];
    // |d| = 0.5 ≤ 1: ½ d² = 0.125
    close(huber.value(&[0.5], &target), 0.125);
    // |d| = 3 > 1: δ (|d| − ½ δ) = 2.5
    close(huber.value(&[3.0], &target), 2.5);
    close(huber.value(&[-3.0], &target), 2.5);
}

#[test]
fn huber_is_continuous_at_the_threshold() {
    let huber = Huber(2.0);
    let target = [0.0_f64];
    let just_inside = huber.value(&[2.0 - 1e-9], &target);
    let just_outside = huber.value(&[2.0 + 1e-9], &target);
    assert!((just_inside - just_outside).abs() < 1e-8);
    let g_inside = huber.gradient(&[2.0 - 1e-9], &target)[0];
    let g_outside = huber.gradient(&[2.0 + 1e-9], &target)[0];
    assert!((g_inside - g_outside).abs() < 1e-8);
}

#[test]
fn huber_gradient_is_capped_at_delta() {
    let huber = Huber(1.5);
    let gradient = huber.gradient(&[0.5_f64, 100.0, -100.0], &[0.0; 3]);
    // d/N im Quadratischen, ±δ/N außerhalb, N = 3
    close(gradient[0], 0.5 / 3.0);
    close(gradient[1], 1.5 / 3.0);
    close(gradient[2], -1.5 / 3.0);
}

#[test]
fn huber_gradient_matches_the_numeric_one() {
    assert_loss_gradient_matches(
        &Huber(1.0),
        [0.3_f64, -2.5, 4.0, 1.1],
        [0.0, 0.0, 0.5, 1.0],
        1e-6,
        1e-7,
    );
}

#[test]
fn huber_with_a_huge_delta_is_half_mse() {
    let output = [0.4_f64, -1.0, 2.0];
    let target = [0.0, 0.5, 1.0];
    close(
        Huber(1000.0).value(&output, &target),
        0.5 * MeanSquaredError.value(&output, &target),
    );
}

#[test]
fn the_default_threshold_is_one() {
    assert_eq!(Huber::default().0, 1.0);
}

#[test]
fn the_new_losses_work_on_any_shape() {
    let output = [[1.0_f32, 2.0], [3.0, 4.0]];
    let target = [[1.0_f32, 0.0], [3.0, 0.0]];
    assert_eq!(MeanAbsoluteError.value(&output, &target), 1.5);
    assert!(Huber(1.0).value(&output, &target) > 0.0);
}

#[cfg(any(feature = "std", feature = "libm"))]
mod softmax_cross_entropy {
    use super::*;
    use neuronws_layers::block::Operation;
    use neuronws_layers::loss::{CategoricalCrossEntropy, SoftmaxCrossEntropy};
    use neuronws_layers::softmax_layer::Softmax;

    fn softmax(logits: [f64; 3]) -> [f64; 3] {
        Softmax::<3, f64>::new().forward(&(), &logits)
    }

    #[test]
    fn it_equals_softmax_followed_by_categorical_cross_entropy() {
        let logits = [1.2, -0.3, 0.7];
        let target = [0.0, 0.0, 1.0];
        let combined = SoftmaxCrossEntropy.value(&logits, &target);
        let two_steps = CategoricalCrossEntropy.value(&softmax(logits), &target);
        assert!((combined - two_steps).abs() < 1e-9);
    }

    #[test]
    fn the_gradient_is_probability_minus_target() {
        let logits = [0.5, 2.0, -1.0];
        let target = [0.0, 1.0, 0.0];
        let gradient = SoftmaxCrossEntropy.gradient(&logits, &target);
        let probabilities = softmax(logits);
        for i in 0..3 {
            close(gradient[i], probabilities[i] - target[i]);
        }
        // Der Gradient summiert sich zu 0, weil beide Verteilungen sich zu 1 summieren.
        close(gradient.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn the_gradient_matches_the_numeric_one() {
        assert_loss_gradient_matches(
            &SoftmaxCrossEntropy,
            [0.5_f64, 2.0, -1.0, 0.1],
            [0.0, 1.0, 0.0, 0.0],
            1e-6,
            1e-7,
        );
        // auch mit einer weichen Verteilung als Ziel
        assert_loss_gradient_matches(
            &SoftmaxCrossEntropy,
            [0.5_f64, 2.0, -1.0],
            [0.2, 0.7, 0.1],
            1e-6,
            1e-7,
        );
    }

    #[test]
    fn huge_logits_do_not_break_it() {
        // Mit Softmax und anschließendem ln käme hier inf oder NaN heraus.
        let logits = [1000.0_f32, -1000.0, 0.0];
        let right = [1.0, 0.0, 0.0];
        let wrong = [0.0, 1.0, 0.0];
        let right_loss = SoftmaxCrossEntropy.value(&logits, &right);
        let wrong_loss = SoftmaxCrossEntropy.value(&logits, &wrong);
        assert!(right_loss.abs() < 1e-6);
        assert!((wrong_loss - 2000.0).abs() < 1e-3);
        for g in SoftmaxCrossEntropy.gradient(&logits, &wrong) {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn it_is_zero_for_a_perfect_prediction_and_positive_otherwise() {
        let target = [0.0_f64, 1.0, 0.0];
        assert!(SoftmaxCrossEntropy.value(&[-30.0, 30.0, -30.0], &target) < 1e-12);
        assert!(SoftmaxCrossEntropy.value(&[0.0, 0.0, 0.0], &target) > 1.0);
        close(
            SoftmaxCrossEntropy.value(&[0.0, 0.0, 0.0], &target),
            3.0_f64.ln(),
        );
    }

    #[test]
    fn it_works_for_any_shape() {
        let logits = [[1.0_f32, 2.0], [0.5, -0.5]];
        let target = [[0.0_f32, 1.0], [0.0, 0.0]];
        assert!(SoftmaxCrossEntropy.value(&logits, &target).is_finite());
        let gradient = SoftmaxCrossEntropy.gradient(&logits, &target);
        assert!(gradient[1][1] > 0.0);
    }

    #[test]
    fn a_net_with_the_combined_loss_learns() {
        use neuronws_layers::activation::Tanh;
        use neuronws_layers::activation_layer::Elementwise;
        use neuronws_layers::chain::LayerExt;
        use neuronws_layers::dense_layer::Linear;
        use neuronws_layers::init::Xavier;
        use neuronws_layers::neurons::neurons;
        use neuronws_layers::optimizer::Sgd;
        use neuronws_layers::random::Rng;
        use neuronws_layers::trainer::Trainer;
        use neuronws_layers::traits::ForwardPath;

        let mut rng = Rng::new(5);
        // Das Netz endet bei den Logits, Softmax steckt in der Fehlerfunktion.
        let mut net = neurons(
            Linear::<2, 6>::new(),
            &Xavier,
            Elementwise::new(Tanh),
            &mut rng,
        )
        .then(neuronws_layers::dense_layer::DenseLayer::<6, 3>::xavier(
            &mut rng,
        ));
        let data = [
            ([2.0_f32, 0.0], [1.0, 0.0, 0.0]),
            ([1.8, 0.3], [1.0, 0.0, 0.0]),
            ([0.0, 2.0], [0.0, 1.0, 0.0]),
            ([0.2, 1.7], [0.0, 1.0, 0.0]),
            ([-2.0, -2.0], [0.0, 0.0, 1.0]),
            ([-1.7, -2.2], [0.0, 0.0, 1.0]),
        ];
        let mut trainer = Trainer::new(SoftmaxCrossEntropy, Sgd::new(0.1), 2);
        let before = trainer.evaluate(&net, &data);
        let after = trainer.fit(&mut net, &data, 200);
        assert!(after < before * 0.1, "{before} -> {after}");

        for (x, target) in &data {
            let logits = net.forward(x);
            let predicted =
                (0..3).fold(0, |best, i| if logits[i] > logits[best] { i } else { best });
            let expected =
                (0..3).fold(0, |best, i| if target[i] > target[best] { i } else { best });
            assert_eq!(predicted, expected);
        }
    }
}
