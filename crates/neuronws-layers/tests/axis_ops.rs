//! Softmax und Standardisierung über eine Achse: Kanäle von Bildern.

use neuronws_layers::block::Block;
use neuronws_layers::gradcheck::assert_gradients_match;
use neuronws_layers::standardize::{ChannelStandardization, Standardize, StandardizeChannels};
use neuronws_layers::traits::{ForwardPath, Trainable};

#[cfg(any(feature = "std", feature = "libm"))]
mod softmax {
    use super::*;
    use neuronws_layers::block::Operation;
    use neuronws_layers::softmax_layer::{
        Softmax, SoftmaxAcross, SoftmaxChannels, SoftmaxChannelsLayer,
    };
    use neuronws_layers::values::Image;

    fn image() -> Image<3, 2, 2, f64> {
        [
            [[0.4, -1.2], [2.0, 0.1]],
            [[1.0, 0.3], [-0.5, 0.0]],
            [[-0.7, 2.2], [0.8, -0.1]],
        ]
    }

    #[test]
    fn every_pixel_gets_its_own_distribution_over_the_channels() {
        let layer = SoftmaxChannelsLayer::<3, 2, 2, f64>::new();
        let output = layer.forward(&image());
        for row in 0..2 {
            for column in 0..2 {
                let sum: f64 = (0..3).map(|c| output[c][row][column]).sum();
                assert!((sum - 1.0).abs() < 1e-12, "Pixel ({row}, {column}): {sum}");
                assert!((0..3).all(|c| output[c][row][column] > 0.0));
            }
        }
    }

    #[test]
    fn it_is_the_same_as_softmax_applied_pixel_by_pixel() {
        let layer = SoftmaxChannelsLayer::<3, 2, 2, f64>::new();
        let output = layer.forward(&image());
        let single = Softmax::<3, f64>::new();
        for row in 0..2 {
            for column in 0..2 {
                let logits = [0, 1, 2].map(|c| image()[c][row][column]);
                let expected = single.forward(&(), &logits);
                for c in 0..3 {
                    assert!((output[c][row][column] - expected[c]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn for_plain_numbers_it_is_the_vector_softmax() {
        let across = SoftmaxAcross::<f64, 4, f64>::new();
        let vector = Softmax::<4, f64>::new();
        let x = [0.3, -1.0, 2.5, 0.0];
        assert_eq!(across.forward(&(), &x), vector.forward(&(), &x));
    }

    #[test]
    fn it_is_not_fooled_by_huge_values() {
        let layer = SoftmaxChannelsLayer::<2, 1, 1>::new();
        let output = layer.forward(&[[[1000.0]], [[1001.0]]]);
        assert!(output.iter().flatten().flatten().all(|p| p.is_finite()));
        assert!((output[0][0][0] + output[1][0][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_gradients_are_correct() {
        assert_gradients_match(
            &SoftmaxChannels::<3, 2, 2, f64>::new(),
            (),
            image(),
            1e-6,
            1e-7,
        );
        assert_gradients_match(
            &SoftmaxChannels::<2, 1, 3>::new(),
            (),
            [[[0.4_f32, -1.2, 2.0]], [[0.1, 0.3, -0.5]]],
            1e-2,
            2e-3,
        );
    }

    #[test]
    fn one_channel_always_gives_probability_one() {
        let layer = SoftmaxChannelsLayer::<1, 2, 2>::new();
        assert_eq!(layer.forward(&[[[3.0, -2.0], [0.0, 9.0]]]), [[[1.0; 2]; 2]]);
    }

    #[test]
    fn it_has_no_parameters_and_works_in_a_chain() {
        assert_eq!(
            <SoftmaxChannelsLayer<3, 4, 4> as Trainable>::PARAMETER_COUNT,
            0
        );
        let _ = Block::<SoftmaxAcross<[f32; 5], 2>>::new();
    }
}

mod channels {
    use super::*;

    fn images() -> [[[[f64; 2]; 1]; 2]; 2] {
        // zwei Beispiele, zwei Kanäle zu 1×2 Pixeln
        [
            [[[9.0, 11.0]], [[990.0, 1010.0]]],
            [[[10.0, 10.0]], [[1000.0, 1000.0]]],
        ]
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn fit_collects_one_statistic_per_channel_over_all_pixels_and_samples() {
        let standardize = StandardizeChannels::<2, 1, 2, f64>::fit(&images());
        // Kanal 0: Werte 9, 11, 10, 10 → Mittelwert 10, Varianz (1 + 1 + 0 + 0)/4 = 0.5
        let [mean0, mean1] = *standardize.mean();
        assert!((mean0 - 10.0).abs() < 1e-12 && (mean1 - 1000.0).abs() < 1e-12);
        let [std0, std1] = *standardize.std_dev();
        // Kanal 1: Werte 990, 1010, 1000, 1000 → Varianz (100 + 100)/4 = 50
        assert!((std0 - 0.5_f64.sqrt()).abs() < 1e-12);
        assert!((std1 - 50.0_f64.sqrt()).abs() < 1e-12);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn the_standardized_training_data_has_mean_zero_and_deviation_one_per_channel() {
        let standardize = StandardizeChannels::<2, 1, 2, f64>::fit(&images());
        let outputs = images().map(|image| standardize.forward(&image));
        for channel in 0..2 {
            let values: Vec<f64> = outputs
                .iter()
                .flat_map(|o| o[channel].iter().flatten().copied())
                .collect();
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            assert!(mean.abs() < 1e-9, "Kanal {channel}: Mittelwert {mean}");
            assert!(
                (variance - 1.0).abs() < 1e-9,
                "Kanal {channel}: Varianz {variance}"
            );
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn with_scalar_channels_it_equals_the_vector_standardize() {
        let data = [[1.0_f64, 1000.0], [2.0, 2000.0], [4.0, 3000.0]];
        let vector = Standardize::<2, f64>::fit(&data);
        let channels = Block::<ChannelStandardization<f64, 2, f64>>::fit(&data);
        assert_eq!(vector.mean(), channels.mean());
        assert_eq!(vector.std_dev(), channels.std_dev());
        assert_eq!(
            vector.forward(&[2.0, 1500.0]),
            channels.forward(&[2.0, 1500.0])
        );
    }

    #[test]
    fn a_constant_channel_is_only_shifted() {
        let layer = Block::<ChannelStandardization<[f64; 2], 2, f64>>::from_statistics(
            [5.0, 0.0],
            [0.0, 2.0],
        );
        assert_eq!(layer.std_dev(), &[1.0, 2.0]);
        assert_eq!(
            layer.forward(&[[6.0, 7.0], [4.0, -2.0]]),
            [[1.0, 2.0], [2.0, -1.0]]
        );
    }

    #[test]
    fn the_gradients_are_correct() {
        let layer = StandardizeChannels::<2, 2, 2, f64>::from_statistics([1.0, -2.0], [2.0, 0.5]);
        let op: &ChannelStandardization<[[f64; 2]; 2], 2, f64> = layer.operation();
        assert_gradients_match(
            op,
            (),
            [[[3.0, 1.0], [0.5, -1.0]], [[2.0, 2.5], [-3.0, 0.0]]],
            1e-6,
            1e-7,
        );
    }

    #[test]
    fn it_has_no_parameters() {
        assert_eq!(
            <StandardizeChannels<3, 8, 8> as Trainable>::PARAMETER_COUNT,
            0
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    #[should_panic(expected = "mindestens ein Beispiel")]
    fn fit_needs_data() {
        let none: [[[f32; 2]; 3]; 0] = [];
        let _ = Block::<ChannelStandardization<[f32; 2], 3>>::fit(&none);
    }
}
