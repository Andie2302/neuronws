//! Faltung mit Schrittweite, Rand und Dilation, 1D-Faltung und transponierte Faltung.

// Die Vergleiche laufen über Kanal-, Zeilen- und Spaltennummern mehrerer Felder.
#![allow(clippy::needless_range_loop)]

use neuronws_layers::block::{Block, Operation};
use neuronws_layers::conv::{
    Conv2d, Conv2dLayer, Conv2dValid, conv_output_size, conv_transpose_output_size, same_padding,
};
use neuronws_layers::conv_transpose::{ConvTranspose2d, ConvTranspose2dLayer};
use neuronws_layers::conv1d::{Conv1d, Conv1dLayer, Conv1dValid};
use neuronws_layers::gradcheck::assert_gradients_match;
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{ForwardPath, Trainable};
use neuronws_layers::values::{self, Image};

// Größen und Makros werden schon beim Prüfen kontrolliert.
neuronws_layers::check_sizes!(
    Conv2d<1, 2, 7, 7, 3, 3, 2, 1, 1, 4, 4>,
    Conv2dValid<1, 4, 6, 6, 3, 3, 4, 4>,
    neuronws_layers::conv_transpose::ConvTranspose2d<1, 1, 3, 3, 3, 3, 2, 0, 1, 7, 7>,
    Conv1d<1, 1, 10, 3, 2, 1, 1, 5>,
);

mod sizes {
    use super::*;

    #[test]
    fn the_output_size_formula_gives_the_known_numbers() {
        assert_eq!(conv_output_size(28, 5, 1, 0, 1), 24);
        assert_eq!(conv_output_size(28, 5, 2, 2, 1), 14);
        assert_eq!(conv_output_size(32, 3, 1, 1, 1), 32);
        assert_eq!(conv_output_size(7, 3, 2, 0, 1), 3);
        // Dilation 2 macht aus dem 3er-Kern einen mit Reichweite 5
        assert_eq!(conv_output_size(9, 3, 1, 0, 2), 5);
        // die Division rundet ab
        assert_eq!(conv_output_size(8, 3, 2, 0, 1), 3);
    }

    #[test]
    fn it_works_in_a_const() {
        const OUT: usize = conv_output_size(28, 5, 2, 2, 1);
        assert_eq!(OUT, 14);
        const UP: usize = conv_transpose_output_size(14, 5, 2, 2, 1);
        assert_eq!(UP, 27);
    }

    #[test]
    fn the_transposed_size_undoes_the_convolution_when_the_division_is_exact() {
        for (input, kernel, stride, padding, dilation) in [
            (7, 3, 2, 0, 1),
            (5, 3, 2, 1, 1),
            (9, 3, 1, 0, 2),
            (11, 5, 1, 2, 1),
        ] {
            let out = conv_output_size(input, kernel, stride, padding, dilation);
            assert_eq!(
                conv_transpose_output_size(out, kernel, stride, padding, dilation),
                input
            );
        }
    }

    #[test]
    fn same_padding_keeps_the_size() {
        assert_eq!(same_padding(1, 1), 0);
        assert_eq!(same_padding(3, 1), 1);
        assert_eq!(same_padding(5, 1), 2);
        assert_eq!(same_padding(3, 2), 2);
        for (kernel, dilation) in [(3, 1), (5, 1), (3, 2), (7, 1), (5, 3)] {
            let padding = same_padding(kernel, dilation);
            assert_eq!(conv_output_size(20, kernel, 1, padding, dilation), 20);
        }
    }

    #[test]
    #[should_panic(expected = "gleichmäßig")]
    fn same_padding_needs_an_odd_reach() {
        let _ = same_padding(4, 1);
    }

    #[test]
    #[should_panic(expected = "mindestens 1")]
    fn a_stride_of_zero_is_refused() {
        let _ = conv_output_size(10, 3, 0, 0, 1);
    }

    #[test]
    #[should_panic(expected = "größer als das Bild")]
    fn a_kernel_larger_than_the_padded_image_is_refused() {
        let _ = conv_output_size(3, 5, 1, 0, 1);
    }

    #[test]
    fn the_macros_write_the_same_types_as_the_long_form() {
        let _: neuronws_layers::conv2d!(in: 1, out: 2, size: (7, 7), kernel: (3, 3), stride: 2, padding: 1) =
            Conv2d::<1, 2, 7, 7, 3, 3, 2, 1, 1, 4, 4>::new();
        let _: neuronws_layers::conv2d!(in: 1, out: 2, size: (6, 6), kernel: (3, 3)) =
            Conv2dValid::<1, 2, 6, 6, 3, 3, 4, 4>::new();
        let _: neuronws_layers::conv2d!(in: 1, out: 2, size: (9, 9), kernel: (3, 3), stride: 1, padding: 0, dilation: 2) =
            Conv2d::<1, 2, 9, 9, 3, 3, 1, 0, 2, 5, 5>::new();
        let _: neuronws_layers::conv2d_layer!(in: 1, out: 2, size: (6, 8), kernel: (3, 3), stride: 3) =
            Conv2dLayer::<1, 2, 6, 8, 3, 3, 3, 0, 1, 2, 2>::new([[[[0.0; 3]; 3]]; 2], [0.0; 2]);
        let _: neuronws_layers::conv1d!(in: 1, out: 2, len: 10, kernel: 3, stride: 2, padding: 1) =
            Conv1d::<1, 2, 10, 3, 2, 1, 1, 5>::new();
        let _: neuronws_layers::conv_transpose2d!(in: 1, out: 1, size: (3, 3), kernel: (3, 3), stride: 2) =
            ConvTranspose2d::<1, 1, 3, 3, 3, 3, 2, 0, 1, 7, 7>::new();
    }
}

/// Ein einfaches, unabhängig geschriebenes Vergleichsprogramm: Bild mit Rand auffüllen und den
/// Kern mit Schrittweite und Dilation darüberschieben.
mod reference {
    pub type Planes = Vec<Vec<Vec<f64>>>;

    pub fn conv2d(
        input: &Planes,
        kernels: &[Planes],
        bias: &[f64],
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Planes {
        let (h, w) = (input[0].len(), input[0][0].len());
        let (kh, kw) = (kernels[0][0].len(), kernels[0][0][0].len());
        let (padded_h, padded_w) = (h + 2 * pad, w + 2 * pad);
        let padded: Planes = input
            .iter()
            .map(|plane| {
                let mut p = vec![vec![0.0; padded_w]; padded_h];
                for r in 0..h {
                    for c in 0..w {
                        p[r + pad][c + pad] = plane[r][c];
                    }
                }
                p
            })
            .collect();
        let out_h = (padded_h - (dilation * (kh - 1) + 1)) / stride + 1;
        let out_w = (padded_w - (dilation * (kw - 1) + 1)) / stride + 1;
        kernels
            .iter()
            .zip(bias)
            .map(|(kernel, &b)| {
                (0..out_h)
                    .map(|r| {
                        (0..out_w)
                            .map(|c| {
                                let mut sum = b;
                                for (i, plane) in padded.iter().enumerate() {
                                    for p in 0..kh {
                                        for q in 0..kw {
                                            sum += kernel[i][p][q]
                                                * plane[r * stride + p * dilation]
                                                    [c * stride + q * dilation];
                                        }
                                    }
                                }
                                sum
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

/// Zahlen, die sich nicht wiederholen und nicht zu glatt sind.
fn pseudo(index: usize) -> f64 {
    ((index * 37 + 11) % 23) as f64 / 11.0 - 1.0
}

fn to_vecs<const A: usize, const B: usize, const C: usize>(
    image: &[[[f64; C]; B]; A],
) -> reference::Planes {
    image
        .iter()
        .map(|plane| plane.iter().map(|row| row.to_vec()).collect())
        .collect()
}

/// Vergleicht `Conv2d` mit dem Vergleichsprogramm und prüft die Gradienten.
fn check_conv2d<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OH: usize,
    const OW: usize,
>() {
    let op = Conv2d::<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, f64>::new();
    let input: Image<C_IN, H, W, f64> = values::from_fn(pseudo);
    let kernels: [[[[f64; KW]; KH]; C_IN]; C_OUT] = values::from_fn(|i| pseudo(i + 5));
    let bias: [f64; C_OUT] = values::from_fn(|i| pseudo(i + 100));

    let output = op.forward(&(kernels, bias), &input);
    let kernel_vecs: Vec<reference::Planes> = kernels.iter().map(to_vecs).collect();
    let expected = reference::conv2d(&to_vecs(&input), &kernel_vecs, &bias, STRIDE, PAD, DIL);

    assert_eq!(expected.len(), C_OUT);
    assert_eq!(expected[0].len(), OH, "Ausgabehöhe");
    assert_eq!(expected[0][0].len(), OW, "Ausgabebreite");
    for o in 0..C_OUT {
        for r in 0..OH {
            for c in 0..OW {
                assert!(
                    (output[o][r][c] - expected[o][r][c]).abs() < 1e-12,
                    "({o}, {r}, {c}): {} gegen {}",
                    output[o][r][c],
                    expected[o][r][c]
                );
            }
        }
    }

    assert_gradients_match(&op, (kernels, bias), input, 1e-6, 1e-7);
}

mod conv2d {
    use super::*;

    #[test]
    fn valid_stays_what_it_was() {
        check_conv2d::<1, 1, 5, 5, 3, 3, 1, 0, 1, 3, 3>();
        check_conv2d::<2, 3, 4, 5, 2, 3, 1, 0, 1, 3, 3>();
    }

    #[test]
    fn stride() {
        check_conv2d::<1, 2, 7, 7, 3, 3, 2, 0, 1, 3, 3>();
        check_conv2d::<2, 1, 8, 6, 3, 2, 2, 0, 1, 3, 3>();
        // Schrittweite größer als der Kern: Pixel dazwischen bleiben unbeachtet
        check_conv2d::<1, 1, 9, 9, 2, 2, 3, 0, 1, 3, 3>();
    }

    #[test]
    fn padding_and_same() {
        check_conv2d::<1, 1, 5, 5, 3, 3, 1, 1, 1, 5, 5>(); // same
        check_conv2d::<2, 2, 6, 4, 5, 3, 1, 2, 1, 6, 6>(); // 5×3-Kern: der Rand gilt für beide Achsen
        check_conv2d::<1, 1, 4, 4, 3, 3, 1, 2, 1, 6, 6>(); // Rand größer als nötig: das Bild wird größer
    }

    #[test]
    fn dilation() {
        check_conv2d::<1, 1, 9, 9, 3, 3, 1, 0, 2, 5, 5>();
        check_conv2d::<1, 2, 7, 7, 2, 2, 1, 0, 3, 4, 4>();
        // same mit Dilation
        check_conv2d::<1, 1, 8, 8, 3, 3, 1, 2, 2, 8, 8>();
    }

    #[test]
    fn everything_together() {
        check_conv2d::<2, 3, 9, 8, 3, 3, 2, 1, 2, 4, 3>();
        check_conv2d::<1, 1, 10, 7, 3, 2, 3, 2, 2, 4, 3>();
    }

    #[test]
    fn wide_rows_use_the_strip_wise_backward_pass_and_agree_with_the_formula() {
        // Ab 16 Ausgabespalten rechnet die Rückwärtsrechnung über Streifen, darunter über jede
        // Ausgabestelle. Beide müssen dasselbe liefern, deshalb hier breite Zeilen mit allen
        // Geometrien: Rand, Schrittweite, Dilation, mehrere Kanäle.
        check_conv2d::<1, 2, 6, 20, 3, 3, 1, 1, 1, 6, 20>(); // same, Zeilenbreite 20
        check_conv2d::<2, 1, 5, 24, 3, 5, 1, 0, 1, 3, 20>(); // valid, nicht quadratischer Kern
        check_conv2d::<1, 2, 7, 40, 3, 3, 2, 1, 2, 3, 19>(); // Schrittweite 2 und Dilation 2
        check_conv2d::<2, 2, 4, 33, 2, 3, 1, 2, 3, 5, 31>(); // großer Rand, Dilation 3
    }

    #[test]
    fn same_padding_keeps_the_image_size_in_the_layer() {
        const P: usize = same_padding(3, 1);
        let layer = Conv2dLayer::<1, 1, 6, 6, 3, 3, 1, P, 1, 6, 6>::new([[[[0.0; 3]; 3]]], [1.0]);
        let output = layer.forward(&[[[0.0; 6]; 6]]);
        assert_eq!(output, [[[1.0; 6]; 6]]);
    }

    #[test]
    fn a_one_hot_kernel_shifts_the_image() {
        // Kern mit einer 1 in der Mitte und Rand 1 ist die Identität ...
        let identity = Conv2dLayer::<1, 1, 3, 3, 3, 3, 1, 1, 1, 3, 3>::new(
            [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]],
            [0.0],
        );
        let image = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]];
        assert_eq!(identity.forward(&image), image);

        // ... mit der 1 oben links verschiebt er alles um eins nach rechts unten.
        let shift = Conv2dLayer::<1, 1, 3, 3, 3, 3, 1, 1, 1, 3, 3>::new(
            [[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            [0.0],
        );
        assert_eq!(
            shift.forward(&image),
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 4.0, 5.0]]]
        );
    }

    #[test]
    fn stride_two_samples_every_other_position() {
        let layer = Conv2dLayer::<1, 1, 5, 5, 1, 1, 2, 0, 1, 3, 3>::new([[[[1.0]]]], [0.0]);
        let image: Image<1, 5, 5> = values::from_fn(|i| i as f32);
        assert_eq!(
            layer.forward(&image),
            [[[0.0, 2.0, 4.0], [10.0, 12.0, 14.0], [20.0, 22.0, 24.0]]]
        );
    }

    #[test]
    fn dilation_spreads_the_kernel_points() {
        // Kern [1, 1] mit Dilation 3 addiert Werte im Abstand 3
        let layer = Conv1dLayer::<1, 1, 6, 2, 1, 0, 3, 3>::new([[[1.0, 1.0]]], [0.0]);
        assert_eq!(
            layer.forward(&[[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]]),
            [[11.0, 22.0, 33.0]]
        );
    }

    #[test]
    fn the_number_of_parameters_does_not_depend_on_the_geometry() {
        assert_eq!(
            <Conv2dLayer<2, 3, 8, 8, 3, 3, 1, 0, 1, 6, 6> as Trainable>::PARAMETER_COUNT,
            <Conv2dLayer<2, 3, 8, 8, 3, 3, 2, 1, 1, 4, 4> as Trainable>::PARAMETER_COUNT
        );
    }

    #[test]
    fn a_strided_padded_layer_recovers_a_teacher_kernel() {
        // Ein "Lehrer" mit festem Kern erzeugt die Zielbilder, der Schüler lernt den Kern wieder.
        type Teacher = Conv2dLayer<1, 1, 6, 6, 3, 3, 2, 1, 1, 3, 3, f64>;
        let teacher_kernel = [[[[0.5, -1.0, 0.25], [2.0, 0.0, -0.5], [1.0, 1.5, -2.0]]]];
        let teacher = Teacher::new(teacher_kernel, [0.0]);

        let data: Vec<(Image<1, 6, 6, f64>, Image<1, 3, 3, f64>)> = (0..12)
            .map(|n| {
                let image: Image<1, 6, 6, f64> = values::from_fn(|i| pseudo(i * 3 + n * 17));
                (image, teacher.forward(&image))
            })
            .collect();

        let mut student = Teacher::new([[[[0.0; 3]; 3]]], [0.0]);
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.1), 4);
        let before = trainer.evaluate(&student, &data);
        let after = trainer.fit(&mut student, &data, 600);
        assert!(after < before * 1e-6, "{before} -> {after}");
        for (learned, expected) in student
            .kernels()
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .zip(teacher_kernel.iter().flatten().flatten().flatten())
        {
            assert!(
                (learned - expected).abs() < 1e-3,
                "{learned} gegen {expected}"
            );
        }
    }
}

mod conv1d {
    use super::*;

    fn check_conv1d<
        const C_IN: usize,
        const C_OUT: usize,
        const L: usize,
        const K: usize,
        const STRIDE: usize,
        const PAD: usize,
        const DIL: usize,
        const OL: usize,
    >() {
        let op = Conv1d::<C_IN, C_OUT, L, K, STRIDE, PAD, DIL, OL, f64>::new();
        let input: [[f64; L]; C_IN] = values::from_fn(pseudo);
        let kernels: [[[f64; K]; C_IN]; C_OUT] = values::from_fn(|i| pseudo(i + 5));
        let bias: [f64; C_OUT] = values::from_fn(|i| pseudo(i + 100));
        let output = op.forward(&(kernels, bias), &input);

        // Die Formel direkt hingeschrieben, mit Positionen als vorzeichenbehaftete Zahlen.
        for (o, channel) in output.iter().enumerate() {
            for (t, &value) in channel.iter().enumerate() {
                let mut sum = bias[o];
                for i in 0..C_IN {
                    for k in 0..K {
                        let position = (t * STRIDE + k * DIL) as isize - PAD as isize;
                        if (0..L as isize).contains(&position) {
                            sum += kernels[o][i][k] * input[i][position as usize];
                        }
                    }
                }
                assert!((value - sum).abs() < 1e-12, "({o}, {t})");
            }
        }
        assert_gradients_match(&op, (kernels, bias), input, 1e-6, 1e-7);
    }

    #[test]
    fn it_matches_the_formula_and_has_correct_gradients() {
        check_conv1d::<1, 1, 8, 3, 1, 0, 1, 6>();
        check_conv1d::<2, 3, 10, 3, 2, 0, 1, 4>();
        check_conv1d::<2, 2, 10, 3, 1, 1, 1, 10>();
        check_conv1d::<1, 2, 12, 3, 1, 0, 3, 6>();
        check_conv1d::<2, 2, 12, 3, 2, 2, 2, 6>();
    }

    #[test]
    fn it_is_the_same_as_conv2d_with_height_one() {
        let kernels1d: [[[f64; 3]; 2]; 2] = values::from_fn(|i| pseudo(i + 3));
        let bias = [0.3, -0.7];
        let signal: [[f64; 9]; 2] = values::from_fn(pseudo);

        // Ein Bild der Höhe 1 mit Kernhöhe 1 verhält sich entlang der Breite wie eine Folge.
        // Rand gäbe es bei Conv2d auch in der Höhe, deshalb Rand 0; Schrittweite und Dilation
        // wirken auf die Höhe 1 nicht und dürfen mitgetestet werden.
        let one_d = Conv1dLayer::<2, 2, 9, 3, 2, 0, 2, 3, f64>::new(kernels1d, bias);
        let two_d = Conv2dLayer::<2, 2, 1, 9, 1, 3, 2, 0, 2, 1, 3, f64>::new(
            kernels1d.map(|per_out| per_out.map(|k| [k])),
            bias,
        );
        let a = one_d.forward(&signal);
        let b = two_d.forward(&signal.map(|s| [s]));
        for o in 0..2 {
            assert_eq!(a[o], b[o][0]);
        }
    }

    #[test]
    fn it_learns_a_smoothing_filter() {
        // Ziel: gleitender Mittelwert über drei Werte
        let teacher = Block::<Conv1dValid<1, 1, 12, 3, 10, f64>>::new([[[1.0 / 3.0; 3]]], [0.0]);
        type Pair = ([[f64; 12]; 1], [[f64; 10]; 1]);
        let data: Vec<Pair> = (0..10)
            .map(|n| {
                let signal: [[f64; 12]; 1] = values::from_fn(|i| pseudo(i * 5 + n * 7));
                (signal, teacher.forward(&signal))
            })
            .collect();
        let mut student = Block::<Conv1dValid<1, 1, 12, 3, 10, f64>>::new([[[0.0; 3]]], [0.0]);
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.2), 5);
        trainer.fit(&mut student, &data, 400);
        for &weight in &student.kernels()[0][0] {
            assert!((weight - 1.0 / 3.0).abs() < 1e-3, "{weight}");
        }
    }

    #[test]
    fn the_number_of_parameters() {
        assert_eq!(
            <Conv1dLayer<2, 4, 100, 5, 2, 2, 1, 50> as Trainable>::PARAMETER_COUNT,
            4 * 2 * 5 + 4
        );
    }
}

mod transposed {
    use super::*;

    #[test]
    fn it_stamps_the_kernel_at_every_input_value() {
        let layer = ConvTranspose2dLayer::<1, 1, 2, 2, 2, 2, 2, 0, 1, 4, 4>::new(
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            [0.5],
        );
        let output = layer.forward(&[[[1.0, 10.0], [100.0, 1000.0]]]);
        assert_eq!(output[0][0], [1.5, 2.5, 10.5, 20.5]);
        assert_eq!(output[0][1], [3.5, 4.5, 30.5, 40.5]);
        assert_eq!(output[0][2], [100.5, 200.5, 1000.5, 2000.5]);
        assert_eq!(output[0][3], [300.5, 400.5, 3000.5, 4000.5]);
    }

    #[test]
    fn overlapping_stamps_add_up() {
        // Schrittweite 1 und Kern 2: Die Stempel überlappen.
        let layer =
            ConvTranspose2dLayer::<1, 1, 2, 2, 2, 2, 1, 0, 1, 3, 3>::new([[[[1.0; 2]; 2]]], [0.0]);
        let output = layer.forward(&[[[1.0, 1.0], [1.0, 1.0]]]);
        assert_eq!(
            output,
            [[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]
        );
    }

    /// `⟨Faltung(x), y⟩ = ⟨x, TransponierteFaltung(y)⟩` für denselben Kern.
    fn adjoint<
        const C_IN: usize,
        const C_OUT: usize,
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
        const PAD: usize,
        const DIL: usize,
        const OH: usize,
        const OW: usize,
    >() {
        let conv = Conv2d::<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, f64>::new();
        let transposed =
            ConvTranspose2d::<C_OUT, C_IN, OH, OW, KH, KW, STRIDE, PAD, DIL, H, W, f64>::new();
        let kernels: [[[[f64; KW]; KH]; C_IN]; C_OUT] = values::from_fn(|i| pseudo(i + 7));
        let x: Image<C_IN, H, W, f64> = values::from_fn(|i| pseudo(i * 3 + 1));
        let y: Image<C_OUT, OH, OW, f64> = values::from_fn(|i| pseudo(i * 5 + 2));

        let forward = conv.forward(&(kernels, [0.0; C_OUT]), &x);
        let backward = transposed.forward(&(kernels, [0.0; C_IN]), &y);
        let left = values::sum(&values::zip_map(&forward, &y, |a, b| a * b));
        let right = values::sum(&values::zip_map(&x, &backward, |a, b| a * b));
        assert!((left - right).abs() < 1e-10, "{left} gegen {right}");
    }

    #[test]
    fn it_is_the_adjoint_of_the_convolution() {
        adjoint::<1, 1, 5, 5, 3, 3, 1, 0, 1, 3, 3>();
        adjoint::<2, 3, 7, 7, 3, 3, 2, 0, 1, 3, 3>();
        adjoint::<1, 2, 5, 5, 3, 3, 2, 1, 1, 3, 3>();
        adjoint::<2, 2, 9, 9, 3, 3, 1, 0, 2, 5, 5>();
        adjoint::<1, 1, 8, 5, 2, 3, 2, 1, 1, 5, 3>();
    }

    fn gradients<
        const C_IN: usize,
        const C_OUT: usize,
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
        const PAD: usize,
        const DIL: usize,
        const OH: usize,
        const OW: usize,
    >() {
        let op = ConvTranspose2d::<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, f64>::new();
        let kernels: [[[[f64; KW]; KH]; C_OUT]; C_IN] = values::from_fn(|i| pseudo(i + 7));
        let bias: [f64; C_OUT] = values::from_fn(|i| pseudo(i + 50));
        let x: Image<C_IN, H, W, f64> = values::from_fn(|i| pseudo(i * 3 + 1));
        assert_gradients_match(&op, (kernels, bias), x, 1e-6, 1e-7);
    }

    #[test]
    fn the_gradients_are_correct() {
        gradients::<1, 1, 3, 3, 2, 2, 2, 0, 1, 6, 6>();
        gradients::<2, 3, 3, 4, 3, 3, 2, 1, 1, 5, 7>();
        gradients::<2, 1, 3, 3, 3, 3, 1, 0, 2, 7, 7>();
        gradients::<1, 2, 4, 4, 3, 3, 3, 1, 1, 10, 10>();
    }

    #[test]
    fn it_upsamples_in_a_chain_with_a_strided_convolution() {
        use neuronws_layers::chain::LayerExt;
        // Encoder verkleinert 8×8 → 4×4, Decoder vergrößert wieder auf 8×8.
        let mut net =
            Conv2dLayer::<1, 2, 8, 8, 2, 2, 2, 0, 1, 4, 4>::new([[[[0.1; 2]; 2]]; 2], [0.0; 2])
                .then(
                    ConvTranspose2dLayer::<2, 1, 4, 4, 2, 2, 2, 0, 1, 8, 8>::new(
                        [[[[0.1; 2]; 2]]; 2],
                        [0.0],
                    ),
                );
        let image: Image<1, 8, 8> = values::from_fn(|i| pseudo(i) as f32);
        let output = net.forward(&image);
        assert_eq!((output[0].len(), output[0][0].len()), (8, 8));

        // Und er lässt sich trainieren: Autoencoder auf wenigen Bildern.
        let data: Vec<(Image<1, 8, 8>, Image<1, 8, 8>)> = (0..4)
            .map(|n| {
                let image: Image<1, 8, 8> = values::from_fn(|i| pseudo(i + n * 13) as f32);
                // einfache, blockweise konstante Bilder lassen sich über 2×2-Blöcke darstellen
                let blocky: Image<1, 8, 8> =
                    values::from_fn(|i| image[0][(i / 8) / 2 * 2][(i % 8) / 2 * 2]);
                (blocky, blocky)
            })
            .collect();
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.02), 4);
        let before = trainer.evaluate(&net, &data);
        let after = trainer.fit(&mut net, &data, 300);
        assert!(after < before * 0.5, "{before} -> {after}");
    }

    #[test]
    fn the_number_of_parameters() {
        assert_eq!(
            <ConvTranspose2dLayer<4, 2, 7, 7, 3, 3, 2, 1, 1, 13, 13> as Trainable>::PARAMETER_COUNT,
            4 * 2 * 3 * 3 + 2
        );
    }
}
