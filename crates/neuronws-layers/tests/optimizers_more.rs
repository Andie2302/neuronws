//! RMSprop, AdaGrad und die Hüllen für Clipping und L2-Regularisierung.

use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::{HasLearningRate, Optimizer, Sgd};
use neuronws_layers::traits::{BackwardPathMut, ForwardPathMut, Trainable};
use neuronws_layers::wrappers::{ClipValue, L2};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-6, "{a} gegen {b}");
}

type Layer1 = DenseLayer<1, 1, f64>;

/// Ein Beispiel mit Eingabe 2 und Gradient 1: grad_w = 2, grad_b = 1.
fn one_gradient_step(layer: &mut Layer1) {
    layer.forward_mut(&[2.0]);
    layer.backward_mut(&[1.0]);
}

fn train_line<O: Optimizer<Layer1>>(optimizer: &mut O, epochs: usize) -> Layer1 {
    let mut layer = Layer1::new([[0.0]], [0.0]);
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x: f64| ([x], [2.0 * x + 1.0]));
    for _ in 0..epochs {
        for (x, target) in &data {
            let y = *layer.forward_mut(x);
            layer.backward_mut(&MeanSquaredError.gradient(&y, target));
        }
        optimizer.step(&mut layer, data.len());
    }
    layer
}

#[cfg(any(feature = "std", feature = "libm"))]
mod adaptive {
    use super::*;
    use neuronws_layers::adaptive::{AdaGrad, RmsProp};
    use neuronws_layers::wrappers::ClipNorm;

    #[test]
    fn rmsprop_follows_the_formula() {
        // g_w = 2, g_b = 1; ρ = 0.9, η = 0.1
        let mut layer = Layer1::new([[1.0]], [0.0]);
        let mut rmsprop = RmsProp::<Layer1>::new(0.1);
        one_gradient_step(&mut layer);
        rmsprop.step(&mut layer, 1);

        // s_w = 0.1 · 4 = 0.4 → Schritt = 0.1 · 2 / √0.4
        close(layer.weights()[0][0], 1.0 - 0.1 * 2.0 / 0.4_f64.sqrt());
        // s_b = 0.1 · 1 → Schritt = 0.1 / √0.1
        close(layer.bias()[0], -0.1 / 0.1_f64.sqrt());

        // zweiter Schritt: s_w = 0.9 · 0.4 + 0.1 · 4 = 0.76
        let before = layer.weights()[0][0];
        one_gradient_step(&mut layer);
        rmsprop.step(&mut layer, 1);
        close(layer.weights()[0][0], before - 0.1 * 2.0 / 0.76_f64.sqrt());
    }

    #[test]
    fn adagrad_follows_the_formula() {
        let mut layer = Layer1::new([[1.0]], [0.0]);
        let mut adagrad = AdaGrad::<Layer1>::new(0.1);
        one_gradient_step(&mut layer);
        adagrad.step(&mut layer, 1);

        // s_w = 4 → Schritt = η g / √s = 0.1 · 2 / 2 = 0.1
        close(layer.weights()[0][0], 1.0 - 0.1);
        close(layer.bias()[0], -0.1);

        // zweiter Schritt: s_w = 8 → Schritt = 0.1 · 2 / √8
        one_gradient_step(&mut layer);
        adagrad.step(&mut layer, 1);
        close(layer.weights()[0][0], 0.9 - 0.2 / 8.0_f64.sqrt());
    }

    #[test]
    fn adagrad_steps_shrink_while_rmsprop_steps_do_not() {
        // Bei immer gleichem Gradienten schrumpft AdaGrad (s wächst), RMSprop pendelt sich ein.
        fn steps<O: Optimizer<Layer1>>(mut optimizer: O) -> Vec<f64> {
            let mut layer = Layer1::new([[0.0]], [0.0]);
            let mut previous = 0.0;
            let mut sizes = vec![];
            for _ in 0..40 {
                one_gradient_step(&mut layer);
                optimizer.step(&mut layer, 1);
                let now = layer.weights()[0][0];
                sizes.push((previous - now).abs());
                previous = now;
            }
            sizes
        }
        let adagrad = steps(AdaGrad::<Layer1>::new(0.1));
        let rmsprop = steps(RmsProp::<Layer1>::new(0.1));
        assert!(adagrad[39] < adagrad[0] / 5.0);
        assert!(rmsprop[39] > rmsprop[0] / 5.0);
    }

    #[test]
    fn both_converge_on_the_line() {
        for layer in [
            train_line(&mut RmsProp::<Layer1>::new(0.02), 1500),
            train_line(&mut AdaGrad::<Layer1>::new(0.5), 1500),
        ] {
            assert!(
                (layer.weights()[0][0] - 2.0).abs() < 0.05,
                "{:?}",
                layer.weights()
            );
            assert!((layer.bias()[0] - 1.0).abs() < 0.05, "{:?}", layer.bias());
        }
    }

    #[test]
    fn both_clear_the_gradients_and_know_their_learning_rate() {
        let mut layer = Layer1::new([[1.0]], [0.0]);
        one_gradient_step(&mut layer);
        let mut rmsprop = RmsProp::<Layer1>::new(0.1);
        rmsprop.step(&mut layer, 1);
        assert_eq!(layer.grad_weights(), &[[0.0]]);

        rmsprop.set_learning_rate(0.3);
        assert_eq!(rmsprop.learning_rate(), 0.3);
        let mut adagrad = AdaGrad::<Layer1>::new(0.1);
        adagrad.set_learning_rate(0.4);
        assert_eq!(adagrad.learning_rate(), 0.4);
    }

    #[test]
    fn custom_settings_and_printing() {
        let rmsprop = RmsProp::<Layer1>::new(0.1).with_decay(0.5, 1e-3);
        assert!(format!("{rmsprop:?}").starts_with("RmsProp"));
        let adagrad = AdaGrad::<Layer1>::new(0.1).with_epsilon(1e-3);
        assert!(format!("{:?}", adagrad.clone()).starts_with("AdaGrad"));
    }

    #[test]
    #[should_panic(expected = "batch_size muss größer als 0 sein")]
    fn a_batch_of_zero_is_refused() {
        let mut layer = Layer1::new([[1.0]], [0.0]);
        RmsProp::<Layer1>::new(0.1).step(&mut layer, 0);
    }

    #[test]
    fn clip_norm_keeps_the_direction_and_shortens_the_length() {
        // g_w = 2, g_b = 1 → Länge √5. Auf 1 begrenzt: (2, 1)/√5.
        let mut layer = Layer1::new([[0.0]], [0.0]);
        one_gradient_step(&mut layer);
        ClipNorm::new(Sgd::new(1.0), 1.0).step(&mut layer, 1);
        let norm = 5.0_f64.sqrt();
        close(layer.weights()[0][0], -2.0 / norm);
        close(layer.bias()[0], -1.0 / norm);
    }

    #[test]
    fn clip_norm_does_nothing_below_the_limit() {
        let mut clipped = Layer1::new([[0.0]], [0.0]);
        let mut plain = Layer1::new([[0.0]], [0.0]);
        one_gradient_step(&mut clipped);
        one_gradient_step(&mut plain);
        ClipNorm::new(Sgd::new(0.1), 10.0).step(&mut clipped, 1);
        Sgd::new(0.1).step(&mut plain, 1);
        assert_eq!(clipped.weights(), plain.weights());
        assert_eq!(clipped.bias(), plain.bias());
    }

    #[test]
    fn clip_norm_looks_at_the_mean_over_the_batch() {
        // Zwei gleiche Beispiele: Summe (4, 2), Mittel (2, 1) mit Länge √5, begrenzt auf 1.
        let mut layer = Layer1::new([[0.0]], [0.0]);
        one_gradient_step(&mut layer);
        one_gradient_step(&mut layer);
        ClipNorm::new(Sgd::new(1.0), 1.0).step(&mut layer, 2);
        let norm = 5.0_f64.sqrt();
        close(layer.weights()[0][0], -2.0 / norm);
        close(layer.bias()[0], -1.0 / norm);
    }

    #[test]
    fn clip_norm_stabilises_a_run_with_a_huge_learning_rate() {
        // Ohne Begrenzung explodiert das Training; mit Begrenzung bleiben die Werte endlich.
        let unclipped = train_line(&mut Sgd::new(5.0), 30);
        let clipped = train_line(&mut ClipNorm::new(Sgd::new(5.0), 0.1), 30);
        let worst = |l: &Layer1| l.weights()[0][0].abs().max(l.bias()[0].abs());
        assert!(!worst(&unclipped).is_finite() || worst(&unclipped) > 1e6);
        assert!(worst(&clipped) < 100.0);
    }

    #[test]
    fn clip_norm_passes_the_learning_rate_through() {
        let mut clip = ClipNorm::new(Sgd::new(0.1_f64), 1.0);
        assert_eq!(clip.learning_rate(), 0.1);
        clip.set_learning_rate(0.2);
        assert_eq!(clip.inner().learning_rate(), 0.2);
        assert_eq!(clip.clone().into_inner().learning_rate(), 0.2);
        let _ = clip.inner_mut();
    }
}

#[test]
fn clip_value_limits_every_gradient_on_its_own() {
    // g_w = 2 wird auf 0.5 begrenzt, g_b = 1 auch: beide Schritte gleich groß.
    let mut layer = Layer1::new([[0.0]], [0.0]);
    one_gradient_step(&mut layer);
    ClipValue::new(Sgd::new(1.0), 0.5).step(&mut layer, 1);
    close(layer.weights()[0][0], -0.5);
    close(layer.bias()[0], -0.5);
}

#[test]
fn clip_value_looks_at_the_mean_over_the_batch() {
    // Vier gleiche Beispiele: Summe (8, 4), Mittel (2, 1), begrenzt auf 1.5 → (1.5, 1).
    let mut layer = Layer1::new([[0.0]], [0.0]);
    for _ in 0..4 {
        one_gradient_step(&mut layer);
    }
    ClipValue::new(Sgd::new(1.0), 1.5).step(&mut layer, 4);
    close(layer.weights()[0][0], -1.5);
    close(layer.bias()[0], -1.0);
}

#[test]
fn clip_value_keeps_small_gradients_and_negative_signs() {
    let mut layer = Layer1::new([[0.0]], [0.0]);
    layer.forward_mut(&[2.0]);
    layer.backward_mut(&[-10.0]); // g_w = −20, g_b = −10
    ClipValue::new(Sgd::new(1.0), 3.0).step(&mut layer, 1);
    close(layer.weights()[0][0], 3.0);
    close(layer.bias()[0], 3.0);
}

#[test]
fn l2_pulls_the_weights_towards_zero_even_without_data_gradient() {
    // Kein Gradient aus den Daten: p ← p − η λ p
    let mut layer = Layer1::new([[2.0]], [-1.0]);
    L2::new(Sgd::new(0.1), 0.5).step(&mut layer, 1);
    close(layer.weights()[0][0], 2.0 - 0.1 * 0.5 * 2.0);
    close(layer.bias()[0], -1.0 + 0.1 * 0.5 * 1.0);
}

#[test]
fn l2_does_not_depend_on_the_batch_size() {
    // Der Zusatz gehört zum Mittelwert, deshalb zieht er bei jeder Batchgröße gleich stark.
    for batch in [1, 4, 32] {
        let mut layer = Layer1::new([[2.0]], [0.0]);
        L2::new(Sgd::new(0.1), 0.5).step(&mut layer, batch);
        close(layer.weights()[0][0], 2.0 - 0.1 * 0.5 * 2.0);
    }
}

#[test]
fn l2_adds_to_the_gradient_of_the_data() {
    let mut layer = Layer1::new([[1.0]], [0.0]);
    one_gradient_step(&mut layer); // g_w = 2
    L2::new(Sgd::new(0.1), 0.25).step(&mut layer, 1);
    close(layer.weights()[0][0], 1.0 - 0.1 * (2.0 + 0.25 * 1.0));
}

#[test]
fn l2_shrinks_the_weights_the_net_ends_up_with() {
    let plain = train_line(&mut Sgd::new(0.05), 400);
    let regularised = train_line(&mut L2::new(Sgd::new(0.05), 0.5), 400);
    assert!(regularised.weights()[0][0].abs() < plain.weights()[0][0].abs());
}

#[test]
fn the_wrappers_can_be_stacked_and_keep_the_learning_rate_reachable() {
    let mut optimizer = L2::new(ClipValue::new(Sgd::new(0.1_f64), 1.0), 0.01);
    assert_eq!(optimizer.learning_rate(), 0.1);
    optimizer.set_learning_rate(0.05);
    assert_eq!(optimizer.inner().inner().learning_rate(), 0.05);

    let layer = train_line(&mut optimizer, 800);
    assert!((layer.weights()[0][0] - 2.0).abs() < 0.2);
}

#[test]
fn a_wrapped_optimizer_works_in_the_trainer_with_a_schedule() {
    use neuronws_layers::schedule::{Constant, Warmup};
    use neuronws_layers::trainer::Trainer;

    let mut layer = Layer1::new([[0.0]], [0.0]);
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x: f64| ([x], [2.0 * x + 1.0]));
    let mut trainer = Trainer::new(MeanSquaredError, ClipValue::new(Sgd::new(0.05), 5.0), 5);
    let before = trainer.evaluate(&layer, &data);
    let after = trainer.fit_scheduled(&mut layer, &data, 200, &Warmup::new(10, Constant));
    assert!(after < before * 0.01);
}

#[test]
fn parameters_after_a_clipped_step_are_a_valid_snapshot_again() {
    // Zur Sicherheit: Die Hüllen lassen die Gradienten genauso zurückgesetzt wie der innere Optimierer.
    let mut layer = Layer1::new([[0.0]], [0.0]);
    one_gradient_step(&mut layer);
    L2::new(ClipValue::new(Sgd::new(0.1), 1.0), 0.1).step(&mut layer, 1);
    assert_eq!(layer.grad_weights(), &[[0.0]]);
    assert_eq!(layer.grad_bias(), &[0.0]);
    let _ = layer.parameters();
}
