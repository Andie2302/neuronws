//! Layer- und Batch-Normalisierung. Beide brauchen die Wurzel.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::gradcheck::assert_gradients_match;
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::norm::{
    BatchNorm, BatchNorm1d, BatchNorm1dLayer, BatchNorm2d, BatchNorm2dLayer, LayerNorm,
    LayerNormLayer,
};
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

fn mean_and_variance(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (
        mean,
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n,
    )
}

mod layer_norm {
    use super::*;

    #[test]
    fn the_output_has_mean_zero_and_unit_variance() {
        let layer = LayerNormLayer::<[f64; 5], f64>::new();
        let y = layer.forward(&[1.0, 5.0, -3.0, 10.0, 2.0]);
        let (mean, variance) = mean_and_variance(&y);
        assert!(mean.abs() < 1e-12, "Mittelwert {mean}");
        assert!((variance - 1.0).abs() < 1e-4, "Varianz {variance}");
    }

    #[test]
    fn gamma_and_beta_scale_and_shift_afterwards() {
        let layer =
            Block::<LayerNorm<[f64; 3], f64>>::with_parameters([2.0, 2.0, 2.0], [10.0, 10.0, 10.0]);
        let plain = LayerNormLayer::<[f64; 3], f64>::new();
        let x = [1.0, 2.0, 6.0];
        let (y, base) = (layer.forward(&x), plain.forward(&x));
        for i in 0..3 {
            assert!((y[i] - (2.0 * base[i] + 10.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn it_does_not_care_about_the_scale_and_offset_of_the_input() {
        let layer = LayerNormLayer::<[f64; 4], f64>::new();
        let x = [0.5, -1.0, 2.0, 3.5];
        let transformed = x.map(|v| 100.0 * v + 7.0);
        let (a, b) = (layer.forward(&x), layer.forward(&transformed));
        for i in 0..4 {
            assert!((a[i] - b[i]).abs() < 1e-3, "{} gegen {}", a[i], b[i]);
        }
    }

    #[test]
    fn constant_input_gives_beta_and_no_nan() {
        let layer = Block::<LayerNorm<[f32; 3]>>::with_parameters([1.0; 3], [0.5, 0.0, -0.5]);
        assert_eq!(layer.forward(&[4.0, 4.0, 4.0]), [0.5, 0.0, -0.5]);
    }

    #[test]
    fn it_starts_as_a_plain_normalisation() {
        let layer = LayerNormLayer::<[f32; 4]>::new();
        assert_eq!(layer.gamma(), &[1.0; 4]);
        assert_eq!(layer.beta(), &[0.0; 4]);
        assert_eq!(<LayerNormLayer<[f32; 4]> as Trainable>::PARAMETER_COUNT, 8);
    }

    #[test]
    fn the_gradients_are_correct_for_vectors_and_images() {
        assert_gradients_match(
            &LayerNorm::<[f64; 4], f64>::new(),
            ([0.5, -1.0, 2.0, 1.0], [0.1, 0.2, -0.3, 0.0]),
            [1.0, -2.0, 0.5, 3.0],
            1e-6,
            1e-6,
        );
        assert_gradients_match(
            &LayerNorm::<[[f64; 2]; 3], f64>::new(),
            (
                [[1.0, 0.5], [-1.0, 2.0], [0.3, 0.7]],
                [[0.0, 0.1], [0.2, -0.1], [0.0, 0.0]],
            ),
            [[1.0, -2.0], [0.5, 3.0], [-1.5, 0.2]],
            1e-6,
            1e-6,
        );
        assert_gradients_match(
            &LayerNorm::<[f32; 3]>::new(),
            ([1.0, 2.0, 0.5], [0.0, 0.1, 0.2]),
            [1.0, -2.0, 0.5],
            1e-2,
            5e-3,
        );
    }

    #[test]
    fn the_gradient_wrt_the_input_sums_to_zero() {
        // Verschiebt man alle Eingaben gleich, ändert sich nichts. Also ist die Summe der
        // Eingabe-Gradienten 0.
        let mut layer = LayerNormLayer::<[f64; 4], f64>::new();
        layer.forward_mut(&[1.0, -2.0, 0.5, 3.0]);
        let grad = *layer.backward_mut(&[0.3, -1.0, 2.0, 0.7]);
        assert!(grad.iter().sum::<f64>().abs() < 1e-10);
    }

    #[test]
    fn it_can_be_trained_inside_a_network() {
        let mut rng = Rng::new(2);
        let mut net = DenseLayer::<2, 4>::xavier(&mut rng)
            .then(LayerNormLayer::<[f32; 4]>::new())
            .then(DenseLayer::<4, 1>::xavier(&mut rng));
        let data = [
            ([0.0_f32, 0.0], [0.0_f32]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [2.0]),
        ];
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 4);
        let before = trainer.evaluate(&net, &data);
        let after = trainer.fit(&mut net, &data, 800);
        assert!(after < before * 0.05, "{before} -> {after}");
    }

    #[test]
    fn a_custom_epsilon_is_used() {
        let tight = LayerNorm::<[f64; 2], f64>::new().with_epsilon(1e-12);
        let loose = LayerNorm::<[f64; 2], f64>::new().with_epsilon(10.0);
        let params = ([1.0, 1.0], [0.0, 0.0]);
        use neuronws_layers::block::Operation;
        let a = tight.forward(&params, &[0.0, 2.0]);
        let b = loose.forward(&params, &[0.0, 2.0]);
        assert!(a[1].abs() > b[1].abs());
    }
}

mod batch_norm {
    use super::*;

    #[test]
    fn at_the_start_it_only_applies_gamma_and_beta_to_the_unnormalised_input() {
        // laufender Mittelwert 0, Varianz 1: y = γ x / √(1 + ε) + β
        let layer = BatchNorm1dLayer::<2, f64>::new();
        let y = layer.forward(&[3.0, -2.0]);
        let inv = 1.0 / (1.0_f64 + 1e-5).sqrt();
        assert!((y[0] - 3.0 * inv).abs() < 1e-12 && (y[1] + 2.0 * inv).abs() < 1e-12);
    }

    #[test]
    fn learning_passes_move_the_running_statistics_but_inference_does_not() {
        let mut layer = BatchNorm1dLayer::<1, f64>::new();
        for _ in 0..10 {
            layer.forward_mut(&[5.0]);
        }
        let after_training = (*layer.running_mean(), *layer.running_var());
        assert!(layer.running_mean()[0] > 3.0);

        for _ in 0..10 {
            layer.forward(&[100.0]);
        }
        assert_eq!(
            (*layer.running_mean(), *layer.running_var()),
            after_training
        );
    }

    #[test]
    fn the_running_statistics_find_mean_and_variance_of_the_data() {
        let mut rng = Rng::new(7);
        let mut layer = BatchNorm1dLayer::<2, f64>::new().with_momentum(0.02);
        for _ in 0..4000 {
            // Kanal 0: Mittelwert 5, Streuung 2. Kanal 1: Mittelwert −3, Streuung 0,5.
            let x = [
                5.0 + 2.0 * f64::from(rng.next_normal()),
                -3.0 + 0.5 * f64::from(rng.next_normal()),
            ];
            layer.forward_mut(&x);
        }
        let [m0, m1] = *layer.running_mean();
        let [v0, v1] = *layer.running_var();
        assert!(
            (m0 - 5.0).abs() < 0.5 && (m1 + 3.0).abs() < 0.15,
            "{m0} {m1}"
        );
        assert!(
            (v0 - 4.0).abs() < 1.0 && (v1 - 0.25).abs() < 0.08,
            "{v0} {v1}"
        );
    }

    #[test]
    fn the_update_follows_the_mixture_formula() {
        // Ein Kanal mit zwei Stellen pro Beispiel: Mittelwert m, Varianz v über die Stellen.
        let mut layer = BatchNorm2dLayer::<1, 1, 2, f64>::new().with_momentum(0.25);
        layer.forward_mut(&[[[1.0, 5.0]]]);
        // m = 3, v = 4. Start μ = 0, σ² = 1, α = 0.25, δ = 3.
        // μ' = 0.75, σ²' = 0.75 · (1 + 0.25 · 9) + 0.25 · 4 = 0.75 · 3.25 + 1 = 3.4375
        assert!((layer.running_mean()[0] - 0.75).abs() < 1e-12);
        assert!((layer.running_var()[0] - 3.4375).abs() < 1e-12);
    }

    #[test]
    fn the_backward_pass_matches_what_forward_calculates() {
        let layer =
            BatchNorm1dLayer::<3, f64>::new().with_statistics([1.0, -2.0, 0.5], [4.0, 0.25, 1.5]);
        let op: &BatchNorm1d<3, f64> = layer.operation();
        assert_gradients_match(
            op,
            ([0.5, -1.0, 2.0], [0.1, 0.2, -0.3]),
            [1.0, -2.0, 0.5],
            1e-6,
            1e-7,
        );
    }

    #[test]
    fn images_get_one_statistic_per_channel() {
        let layer =
            BatchNorm2dLayer::<2, 2, 2, f64>::new().with_statistics([1.0, 10.0], [4.0, 9.0]);
        let op: &BatchNorm2d<2, 2, 2, f64> = layer.operation();
        assert_gradients_match(
            op,
            ([0.5, -1.0], [0.1, 0.2]),
            [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 11.0], [9.0, 12.0]]],
            1e-6,
            1e-7,
        );
        // Kanal 0: (x − 1)/2, Kanal 1: (x − 10)/3 (bis auf ε)
        let y = layer.forward(&[[[3.0, 3.0], [3.0, 3.0]], [[13.0, 13.0], [13.0, 13.0]]]);
        assert!((y[0][0][0] - 1.0).abs() < 1e-4);
        assert!((y[1][1][1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn the_image_statistics_pool_all_pixels_of_a_channel() {
        let mut layer = BatchNorm2dLayer::<1, 2, 2, f64>::new().with_momentum(1.0);
        // Mit Momentum 1 ersetzt ein Beispiel die Statistik ganz: Mittelwert und Varianz der 4 Pixel.
        layer.forward_mut(&[[[1.0, 2.0], [3.0, 6.0]]]);
        assert!((layer.running_mean()[0] - 3.0).abs() < 1e-12);
        assert!((layer.running_var()[0] - 3.5).abs() < 1e-12);
    }

    #[test]
    fn gamma_and_beta_are_learnable_and_the_statistics_are_not() {
        assert_eq!(<BatchNorm1dLayer<4> as Trainable>::PARAMETER_COUNT, 8);
        assert_eq!(<BatchNorm2dLayer<3, 8, 8> as Trainable>::PARAMETER_COUNT, 6);
        let mut layer = BatchNorm1dLayer::<2, f64>::new();
        layer.forward_mut(&[1.0, 2.0]);
        let before = layer.parameters();
        layer.backward_mut(&[1.0, 1.0]);
        assert_eq!(layer.parameters(), before);
    }

    #[test]
    fn gamma_and_beta_receive_gradients() {
        let mut layer = BatchNorm1dLayer::<1, f64>::new().with_statistics([2.0], [4.0]);
        layer.forward_mut(&[2.0]); // Training verschiebt die Statistik minimal, das ist hier egal
        layer.backward_mut(&[3.0]);
        let mut grads = vec![];
        layer.inspect_parameters(&mut |_, g| grads.push(g));
        // ∂L/∂β = g = 3, ∂L/∂γ = g · x̂ mit x̂ ≈ 0
        assert!((grads[1] - 3.0).abs() < 1e-12);
        assert!(grads[0].abs() < 0.5);
    }

    #[test]
    fn a_network_with_batch_norm_learns_a_task_with_badly_scaled_inputs() {
        // Eingaben um 100 herum: ohne Normalisierung sättigen die Neuronen sofort.
        let mut rng = Rng::new(11);
        let make = |rng: &mut Rng| {
            let a = rng.uniform(-1.0, 1.0);
            let b = rng.uniform(-1.0, 1.0);
            ([100.0 + 5.0 * a, 200.0 + 5.0 * b], [a + b])
        };
        let data: Vec<([f32; 2], [f32; 1])> = (0..64).map(|_| make(&mut rng)).collect();

        let mut net = BatchNorm1dLayer::<2>::new()
            .with_statistics([100.0, 200.0], [25.0 / 3.0, 25.0 / 3.0])
            .then(DenseLayer::<2, 1>::xavier(&mut rng));
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.02), 8);
        let before = trainer.evaluate(&net, &data);
        let after = trainer.fit(&mut net, &data, 150);
        assert!(after < before * 0.2, "{before} -> {after}");
        assert!(after < 0.1);
    }

    #[test]
    fn the_type_aliases_are_the_same_thing() {
        let _: BatchNorm<f32, 4> = *Block::<BatchNorm<f32, 4>>::new().operation();
        let _: BatchNorm1d<4> = *BatchNorm1dLayer::<4>::new().operation();
    }
}
