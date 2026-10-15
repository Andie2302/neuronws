//! Normalverteilte Zufallszahlen (Box-Muller) und die Startwerte, die darauf aufbauen.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::block::Block;
use neuronws_layers::conv::Conv2d;
use neuronws_layers::dense_layer::{DenseLayer, Linear};
use neuronws_layers::init::{HeNormal, Initializer, Normal, Orthogonal, Uniform, XavierNormal};
use neuronws_layers::random::Rng;
use neuronws_layers::scalar::Scalar;

fn mean_and_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

#[test]
fn next_normal_has_mean_zero_and_deviation_one() {
    let mut rng = Rng::new(1);
    let values: Vec<f64> = (0..40_000).map(|_| f64::from(rng.next_normal())).collect();
    let (mean, std) = mean_and_std(&values);
    assert!(mean.abs() < 0.03, "Mittelwert {mean}");
    assert!((std - 1.0).abs() < 0.03, "Streuung {std}");
}

#[test]
fn next_normal_follows_the_bell_curve() {
    let mut rng = Rng::new(2);
    let values: Vec<f32> = (0..40_000).map(|_| rng.next_normal()).collect();
    let within = |k: f32| values.iter().filter(|v| v.abs() <= k).count() as f64 / 40_000.0;
    // 68–95–99,7-Regel
    assert!((within(1.0) - 0.6827).abs() < 0.015, "{}", within(1.0));
    assert!((within(2.0) - 0.9545).abs() < 0.01, "{}", within(2.0));
    assert!((within(3.0) - 0.9973).abs() < 0.005, "{}", within(3.0));
    assert!(values.iter().all(|v| v.is_finite()));
}

#[test]
fn next_normal_is_deterministic_and_uses_both_numbers_of_a_pair() {
    let mut a = Rng::new(7);
    let mut b = Rng::new(7);
    for _ in 0..10 {
        assert_eq!(a.next_normal(), b.next_normal());
    }

    // Zwei Normalzahlen kosten genau zwei gleichverteilte Zahlen.
    let mut normal = Rng::new(3);
    let mut uniform = Rng::new(3);
    normal.next_normal();
    normal.next_normal();
    uniform.next_f32();
    uniform.next_f32();
    assert_eq!(normal.next_u32(), uniform.next_u32());
}

fn sample_std<I: Initializer>(initializer: &I, fan_in: usize, fan_out: usize) -> f64 {
    let mut rng = Rng::new(11);
    let values: Vec<f64> = (0..30_000)
        .map(|_| initializer.sample::<f64>(&mut rng, fan_in, fan_out))
        .collect();
    let (mean, std) = mean_and_std(&values);
    assert!(
        mean.abs() < 0.05 * std.max(0.01) + 0.01,
        "Mittelwert {mean}"
    );
    std
}

#[test]
fn the_normal_initializers_use_the_documented_deviation() {
    assert!((sample_std(&Normal(0.1), 5, 7) - 0.1).abs() < 0.003);
    let xavier = (2.0_f64 / (30.0 + 10.0)).sqrt();
    assert!((sample_std(&XavierNormal, 30, 10) - xavier).abs() < 0.1 * xavier);
    let he = (2.0_f64 / 50.0).sqrt();
    assert!((sample_std(&HeNormal, 50, 3) - he).abs() < 0.1 * he);
}

#[test]
fn the_normal_initializers_work_for_f32_and_f64() {
    let mut rng = Rng::new(4);
    let a: f32 = XavierNormal.sample(&mut rng, 4, 4);
    let b: f64 = HeNormal.sample(&mut rng, 4, 4);
    assert!(a.is_finite() && b.is_finite());
}

#[test]
fn the_default_fill_draws_value_by_value_like_before() {
    // Der Weg über `fill` darf die Zufallsfolge der bisherigen Schichten nicht verändern.
    let layer = Block::initialized(Linear::<3, 2>::new(), &Uniform(1.0), &mut Rng::new(9));
    let mut rng = Rng::new(9);
    for row in layer.weights() {
        for &w in row {
            assert_eq!(w, Uniform(1.0).sample::<f32>(&mut rng, 3, 2));
        }
    }
}

/// Skalarprodukt der Zeilen `a` und `b` einer Matrix.
fn row_dot<T: Scalar, const R: usize, const C: usize>(m: &[[T; C]; R], a: usize, b: usize) -> f64 {
    (0..C).map(|k| (m[a][k] * m[b][k]).to_f64()).sum()
}

fn column_dot<T: Scalar, const R: usize, const C: usize>(
    m: &[[T; C]; R],
    a: usize,
    b: usize,
) -> f64 {
    (0..R).map(|k| (m[k][a] * m[k][b]).to_f64()).sum()
}

#[test]
fn orthogonal_rows_are_perpendicular_and_have_unit_length() {
    // weniger Zeilen als Spalten: die Zeilen sind orthonormal
    let layer = Block::initialized(Linear::<7, 4>::new(), &Orthogonal(1.0), &mut Rng::new(1));
    let w = layer.weights();
    for a in 0..4 {
        for b in 0..4 {
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!(
                (row_dot(w, a, b) - expected).abs() < 1e-5,
                "Zeilen {a}·{b} = {}",
                row_dot(w, a, b)
            );
        }
    }
}

#[test]
fn orthogonal_columns_are_orthonormal_when_there_are_more_rows() {
    // mehr Zeilen als Spalten: dann sind die Spalten orthonormal
    let layer: DenseLayer<3, 6, f64> = Block::initialized(
        Linear::<3, 6, f64>::new(),
        &Orthogonal(1.0),
        &mut Rng::new(2),
    );
    let w = layer.weights();
    for a in 0..3 {
        for b in 0..3 {
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!((column_dot(w, a, b) - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn a_square_orthogonal_matrix_is_orthogonal_both_ways() {
    let layer: DenseLayer<5, 5, f64> = Block::initialized(
        Linear::<5, 5, f64>::new(),
        &Orthogonal(1.0),
        &mut Rng::new(3),
    );
    let w = layer.weights();
    for a in 0..5 {
        for b in 0..5 {
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!((row_dot(w, a, b) - expected).abs() < 1e-12);
            assert!((column_dot(w, a, b) - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn the_gain_scales_the_length() {
    let layer: DenseLayer<6, 3, f64> = Block::initialized(
        Linear::<6, 3, f64>::new(),
        &Orthogonal(2.0),
        &mut Rng::new(4),
    );
    let w = layer.weights();
    for a in 0..3 {
        assert!((row_dot(w, a, a) - 4.0).abs() < 1e-12);
    }
    assert!(row_dot(w, 0, 1).abs() < 1e-12);
    assert_eq!(Orthogonal::default().0, 1.0);
}

#[test]
fn orthogonal_is_deterministic_and_different_per_seed() {
    let make =
        |seed| Block::initialized(Linear::<4, 4>::new(), &Orthogonal(1.0), &mut Rng::new(seed));
    assert_eq!(make(5).weights(), make(5).weights());
    assert_ne!(make(5).weights(), make(6).weights());
}

#[test]
fn convolution_kernels_are_orthogonal_per_output_channel() {
    // Jeder Ausgabekanal ist eine Zeile der Länge C_IN·KH·KW = 18.
    let conv = Block::initialized(
        Conv2d::<2, 3, 5, 5, 3, 3, 3, 3, f64>::new(),
        &Orthogonal(1.0),
        &mut Rng::new(6),
    );
    let flat = |o: usize| -> Vec<f64> {
        conv.kernels()[o]
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect()
    };
    for a in 0..3 {
        for b in 0..3 {
            let dot: f64 = flat(a).iter().zip(flat(b)).map(|(x, y)| x * y).sum();
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-12, "Kanäle {a}·{b} = {dot}");
        }
    }
}

#[test]
fn the_sample_fallback_has_the_scale_of_one_over_root_fan_in() {
    let std = sample_std(&Orthogonal(1.0), 16, 4);
    assert!((std - 0.25).abs() < 0.01);
}
