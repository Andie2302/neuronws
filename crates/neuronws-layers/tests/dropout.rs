//! Dropout, Gaußsches Rauschen und die Reparametrisierung: Zufall, der nur beim Lernen wirkt.

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::dropout::{Dropout, DropoutLayer};
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

mod dropout {
    use super::*;

    #[test]
    fn at_inference_it_does_nothing() {
        let layer = DropoutLayer::<[f32; 4]>::new(0.9, 1);
        let x = [1.0, 2.0, 3.0, 4.0];
        for _ in 0..5 {
            assert_eq!(layer.forward(&x), x);
        }
    }

    #[test]
    fn while_learning_values_are_dropped_or_scaled_up() {
        let mut layer = DropoutLayer::<[f64; 1000], f64>::new(0.25, 3);
        let out = *layer.forward_mut(&[1.0; 1000]);
        let scale = 1.0 / 0.75;
        assert!(out.iter().all(|&v| v == 0.0 || (v - scale).abs() < 1e-12));
        let kept = out.iter().filter(|&&v| v != 0.0).count();
        assert!((700..=800).contains(&kept), "{kept} behalten");
    }

    #[test]
    fn the_average_stays_the_same() {
        // Das Hochskalieren hält den Erwartungswert bei 1.
        let mut layer = DropoutLayer::<[f64; 4000], f64>::new(0.4, 5);
        let out = *layer.forward_mut(&[1.0; 4000]);
        let mean = out.iter().sum::<f64>() / 4000.0;
        assert!((mean - 1.0).abs() < 0.05, "Mittelwert {mean}");
    }

    #[test]
    fn a_new_mask_is_drawn_for_every_learning_pass() {
        let mut layer = DropoutLayer::<[f32; 64]>::new(0.5, 9);
        let a = *layer.forward_mut(&[1.0; 64]);
        let b = *layer.forward_mut(&[1.0; 64]);
        assert_ne!(a, b);
    }

    #[test]
    fn the_gradient_only_flows_through_the_kept_values() {
        let mut layer = DropoutLayer::<[f64; 200], f64>::new(0.5, 4);
        let out = *layer.forward_mut(&[2.0; 200]);
        let grad = *layer.backward_mut(&[1.0; 200]);
        for i in 0..200 {
            if out[i] == 0.0 {
                assert_eq!(grad[i], 0.0);
            } else {
                // Eingabe 2 wurde zu 4 (Faktor 2), also fließt der Gradient mit Faktor 2 zurück.
                assert_eq!(out[i], 4.0);
                assert_eq!(grad[i], 2.0);
            }
        }
    }

    #[test]
    fn the_backward_pass_matches_the_derivative_of_the_mask() {
        // Bei fester Maske ist die Funktion x ↦ x ⊙ m, ihre Ableitung ist m.
        let mut layer = DropoutLayer::<[f64; 8], f64>::new(0.5, 21);
        let x = [0.5, -1.0, 2.0, 3.0, 0.1, -0.2, 4.0, 1.5];
        let out = *layer.forward_mut(&x);
        let grad_out = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let grad = *layer.backward_mut(&grad_out);
        let mask = *layer.operation().mask();
        for i in 0..8 {
            assert_eq!(out[i], x[i] * mask[i]);
            assert_eq!(grad[i], grad_out[i] * mask[i]);
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_masks() {
        let run = |seed| {
            let mut layer = DropoutLayer::<[f32; 32]>::new(0.5, seed);
            (
                *layer.forward_mut(&[1.0; 32]),
                *layer.forward_mut(&[1.0; 32]),
            )
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn reseeding_repeats_a_run() {
        let mut layer = DropoutLayer::<[f32; 32]>::new(0.5, 1);
        let first = *layer.forward_mut(&[1.0; 32]);
        layer.forward_mut(&[1.0; 32]);
        layer.operation_mut().reseed(1);
        assert_eq!(*layer.forward_mut(&[1.0; 32]), first);
    }

    #[test]
    fn a_rate_of_zero_changes_nothing_even_while_learning() {
        let mut layer = DropoutLayer::<[f32; 5]>::new(0.0, 1);
        let x = [1.0, -2.0, 3.0, 0.0, 5.0];
        assert_eq!(*layer.forward_mut(&x), x);
        assert_eq!(*layer.backward_mut(&x), x);
    }

    #[test]
    #[should_panic(expected = "rate muss in [0, 1) liegen")]
    fn a_rate_of_one_is_refused() {
        let _ = Dropout::<[f32; 3]>::new(1.0, 1);
    }

    #[test]
    #[should_panic(expected = "rate muss in [0, 1) liegen")]
    fn a_negative_rate_is_refused() {
        let _ = Dropout::<[f32; 3]>::new(-0.1, 1);
    }

    #[test]
    fn it_has_no_parameters_and_can_be_cloned_and_printed() {
        assert_eq!(<DropoutLayer<[f32; 4]> as Trainable>::PARAMETER_COUNT, 0);
        let layer = DropoutLayer::<[f32; 4]>::new(0.3, 1);
        assert_eq!(layer.operation().rate(), 0.3);
        assert!(format!("{:?}", layer.clone()).starts_with("Block"));
    }

    #[test]
    fn it_works_on_any_shape() {
        let mut layer = DropoutLayer::<[[f32; 3]; 4]>::new(0.5, 8);
        let out = *layer.forward_mut(&[[1.0; 3]; 4]);
        assert!(out.iter().flatten().all(|&v| v == 0.0 || v == 2.0));
    }

    #[test]
    fn evaluation_ignores_dropout_but_training_uses_it() {
        let mut rng = Rng::new(3);
        let mut net = DenseLayer::<2, 16>::random(&mut rng, 0.5)
            .then(ActivationLayer::<16, _>::new(Relu))
            .then(DropoutLayer::<[f32; 16]>::new(0.5, 17))
            .then(DenseLayer::<16, 1>::random(&mut rng, 0.5));
        let input = [0.7, -0.4];
        // Der Einsatz ist deterministisch, trotz Dropout mitten im Netz.
        assert_eq!(net.forward(&input), net.forward(&input));
        // Beim Lernen streut das Ergebnis, weil jedes Mal andere Neuronen fehlen.
        let a = *net.forward_mut(&input);
        let b = *net.forward_mut(&input);
        assert_ne!(a, b);
    }

    #[test]
    fn a_network_with_dropout_still_learns() {
        let mut rng = Rng::new(5);
        let mut net = DenseLayer::<1, 16>::random(&mut rng, 0.5)
            .then(ActivationLayer::<16, _>::new(Relu))
            .then(DropoutLayer::<[f32; 16]>::new(0.2, 23))
            .then(DenseLayer::<16, 1>::random(&mut rng, 0.5));
        let data: Vec<([f32; 1], [f32; 1])> = (0..20)
            .map(|i| {
                let x = -1.0 + 0.1 * i as f32;
                ([x], [2.0 * x + 0.5])
            })
            .collect();
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.02), 4);
        let before = trainer.evaluate(&net, &data);
        let after = trainer.fit(&mut net, &data, 300);
        let evaluated = trainer.evaluate(&net, &data);
        assert!(evaluated < before * 0.1, "{before} -> {evaluated}");
        // Das Netz ohne Dropout (Einsatz) ist besser als die verrauschte Lernrunde.
        assert!(evaluated <= after * 1.5 + 0.05);
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
mod noise {
    use super::*;
    use neuronws_layers::dropout::{GaussianNoise, GaussianNoiseLayer};

    #[test]
    fn at_inference_there_is_no_noise() {
        let layer = GaussianNoiseLayer::<[f32; 3]>::new(1.0, 1);
        assert_eq!(layer.forward(&[1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn while_learning_the_noise_has_the_requested_deviation() {
        let mut layer = GaussianNoiseLayer::<[f64; 4000], f64>::new(0.5, 3);
        let out = *layer.forward_mut(&[10.0; 4000]);
        let noise: Vec<f64> = out.iter().map(|v| v - 10.0).collect();
        let mean = noise.iter().sum::<f64>() / 4000.0;
        let std = (noise.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4000.0).sqrt();
        assert!(mean.abs() < 0.05, "Mittelwert {mean}");
        assert!((std - 0.5).abs() < 0.03, "Streuung {std}");
    }

    #[test]
    fn the_gradient_passes_through_unchanged() {
        let mut layer = GaussianNoiseLayer::<[f32; 3]>::new(1.0, 1);
        layer.forward_mut(&[1.0, 2.0, 3.0]);
        assert_eq!(*layer.backward_mut(&[0.5, -1.0, 2.0]), [0.5, -1.0, 2.0]);
    }

    #[test]
    fn zero_deviation_is_the_identity_and_the_seed_makes_it_repeatable() {
        let mut quiet = GaussianNoiseLayer::<[f32; 3]>::new(0.0, 1);
        assert_eq!(*quiet.forward_mut(&[1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);

        let run = |seed| *GaussianNoiseLayer::<[f32; 3]>::new(1.0, seed).forward_mut(&[0.0; 3]);
        assert_eq!(run(4), run(4));
        assert_ne!(run(4), run(5));

        let mut layer = GaussianNoiseLayer::<[f32; 3]>::new(1.0, 4);
        let first = *layer.forward_mut(&[0.0; 3]);
        layer.operation_mut().reseed(4);
        assert_eq!(*layer.forward_mut(&[0.0; 3]), first);
    }

    #[test]
    #[should_panic(expected = "std_dev darf nicht negativ sein")]
    fn a_negative_deviation_is_refused() {
        let _ = GaussianNoise::<[f32; 2]>::new(-1.0, 1);
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
mod reparameterize {
    use super::*;
    use neuronws_layers::dropout::ReparameterizeLayer;

    type Pair = ([f64; 3], [f64; 3]);

    #[test]
    fn at_inference_it_returns_the_mean() {
        let layer = ReparameterizeLayer::<[f64; 3], f64>::new(1);
        let input: Pair = ([1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
        assert_eq!(layer.forward(&input), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn while_learning_z_is_mean_plus_sigma_times_noise() {
        let mut layer = ReparameterizeLayer::<[f64; 3], f64>::new(2);
        let (mean, log_var): Pair = ([1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
        let z = *layer.forward_mut(&(mean, log_var));
        let noise = *layer.operation().noise();
        for i in 0..3 {
            let sigma = (0.5 * log_var[i]).exp();
            assert!((z[i] - (mean[i] + sigma * noise[i])).abs() < 1e-12);
        }
    }

    #[test]
    fn the_gradients_are_the_derivatives_for_the_drawn_noise() {
        let mut layer = ReparameterizeLayer::<[f64; 3], f64>::new(3);
        let (mean, log_var): Pair = ([1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
        layer.forward_mut(&(mean, log_var));
        let noise = *layer.operation().noise();
        let g = [1.0, -2.0, 0.5];
        let (grad_mean, grad_log_var) = *layer.backward_mut(&g);

        // z(μ, v) = μ + e^(v/2) · ε bei festem ε, numerisch abgeleitet
        let z = |mean: &[f64; 3], log_var: &[f64; 3]| -> f64 {
            (0..3)
                .map(|i| g[i] * (mean[i] + (0.5 * log_var[i]).exp() * noise[i]))
                .sum()
        };
        let h = 1e-6;
        for i in 0..3 {
            let (mut up, mut down) = (mean, mean);
            up[i] += h;
            down[i] -= h;
            let numeric = (z(&up, &log_var) - z(&down, &log_var)) / (2.0 * h);
            assert!((grad_mean[i] - numeric).abs() < 1e-6);

            let (mut up, mut down) = (log_var, log_var);
            up[i] += h;
            down[i] -= h;
            let numeric = (z(&mean, &up) - z(&mean, &down)) / (2.0 * h);
            assert!((grad_log_var[i] - numeric).abs() < 1e-6);
        }
    }

    #[test]
    fn the_samples_have_the_requested_mean_and_deviation() {
        let mut layer = ReparameterizeLayer::<[f64; 1], f64>::new(4);
        let input = ([3.0], [2.0_f64.ln() * 2.0]); // σ = e^(ln 2) = 2
        let zs: Vec<f64> = (0..5000).map(|_| layer.forward_mut(&input)[0]).collect();
        let mean = zs.iter().sum::<f64>() / 5000.0;
        let std = (zs.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / 5000.0).sqrt();
        assert!((mean - 3.0).abs() < 0.1, "Mittelwert {mean}");
        assert!((std - 2.0).abs() < 0.1, "Streuung {std}");
    }

    #[test]
    fn a_zero_variance_leaves_the_mean_unchanged_apart_from_epsilon_one() {
        // log σ² → −∞ heißt σ → 0, dann ist z = μ.
        let mut layer = ReparameterizeLayer::<[f64; 2], f64>::new(5);
        let z = *layer.forward_mut(&([1.0, -1.0], [-200.0, -200.0]));
        assert!((z[0] - 1.0).abs() < 1e-30 + 1e-12 && (z[1] + 1.0).abs() < 1e-12);
    }
}
