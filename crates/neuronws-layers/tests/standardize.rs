use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::scalar::Scalar;
use neuronws_layers::standardize::Standardize;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

fn assert_close(a: f32, b: f32, tolerance: f32) {
    assert!((a - b).abs() < tolerance, "{a} gegen {b}");
}

#[test]
fn from_statistics_subtracts_the_mean_and_divides_by_the_deviation() {
    let layer = Standardize::from_statistics([10.0_f32, 0.0], [2.0, 4.0]);
    assert_eq!(layer.forward(&[14.0, 8.0]), [2.0, 2.0]);
    assert_eq!(layer.forward(&[10.0, 0.0]), [0.0, 0.0]);
}

#[test]
fn a_deviation_of_zero_is_treated_as_one_and_never_divides_by_zero() {
    let layer = Standardize::from_statistics([5.0_f32, 5.0, 5.0], [0.0, -1.0, f32::NAN]);
    assert_eq!(layer.std_dev(), &[1.0, 1.0, 1.0]);
    let out = layer.forward(&[7.0, 7.0, 7.0]);
    assert!(out.iter().all(|v| v.is_finite()));
    assert_eq!(out, [2.0, 2.0, 2.0]);
}

#[test]
fn backward_divides_by_the_deviation_and_matches_the_numeric_gradient() {
    let mut layer = Standardize::from_statistics([1.0_f32, -2.0], [2.0, 0.5]);
    let x = [3.0_f32, 1.0];
    let weights = [0.7_f32, -1.3];
    // Beliebiger Verlust darüber: L = Σ w[i] · y[i]
    let loss = |x: [f32; 2]| {
        layer_forward(&Standardize::from_statistics([1.0, -2.0], [2.0, 0.5]), x)
            .iter()
            .zip(weights)
            .map(|(y, w)| y * w)
            .sum::<f32>()
    };
    layer.forward_mut(&x);
    let analytic = *layer.backward_mut(&weights);

    for (i, a) in analytic.iter().enumerate() {
        let (mut up, mut down) = (x, x);
        up[i] += 1e-2;
        down[i] -= 1e-2;
        assert_close(*a, (loss(up) - loss(down)) / 2e-2, 1e-3);
    }
}

fn layer_forward(layer: &Standardize<2>, x: [f32; 2]) -> [f32; 2] {
    layer.forward(&x)
}

#[test]
fn it_has_no_parameters_to_learn() {
    assert_eq!(<Standardize<5> as Trainable>::PARAMETER_COUNT, 0);
    let mut layer = Standardize::from_statistics([0.0_f32; 3], [1.0; 3]);
    let mut visited = 0;
    layer.visit_parameters(&mut |_, _| visited += 1);
    assert_eq!(visited, 0);
}

#[test]
fn it_chains_in_front_of_other_layers() {
    let net = Standardize::from_statistics([10.0_f32], [5.0]).then(DenseLayer::new([[2.0]], [1.0]));
    assert_eq!(net.forward(&[20.0]), [5.0]); // (20 − 10) / 5 = 2, dann 2·2 + 1
}

#[test]
fn it_works_with_f64() {
    let layer = Standardize::from_statistics([1.0_f64], [2.0]);
    assert_eq!(layer.forward(&[5.0]), [2.0]);
    assert_eq!(<f64 as Scalar>::ONE, 1.0);
}

#[cfg(any(feature = "std", feature = "libm"))]
mod fit {
    use super::*;

    #[test]
    fn fit_computes_mean_and_population_deviation() {
        let layer = Standardize::fit(&[[1.0_f32], [2.0], [3.0], [4.0]]);
        assert_close(layer.mean()[0], 2.5, 1e-6);
        // Varianz = (2,25 + 0,25 + 0,25 + 2,25) / 4 = 1,25
        assert_close(layer.std_dev()[0], 1.25_f32.sqrt(), 1e-6);
    }

    #[test]
    fn standardized_data_has_mean_zero_and_deviation_one() {
        let data: Vec<[f32; 2]> = (0..200)
            .map(|i| [i as f32 * 0.37 + 5.0, 1000.0 - (i as f32).powi(2) * 0.05])
            .collect();
        let layer = Standardize::fit(&data);
        let out: Vec<[f32; 2]> = data.iter().map(|x| layer.forward(x)).collect();

        for column in 0..2 {
            let mean = out.iter().map(|x| x[column]).sum::<f32>() / out.len() as f32;
            let var =
                out.iter().map(|x| (x[column] - mean).powi(2)).sum::<f32>() / out.len() as f32;
            assert_close(mean, 0.0, 1e-3);
            assert_close(var.sqrt(), 1.0, 1e-3);
        }
    }

    #[test]
    fn the_result_does_not_depend_on_the_units_of_the_data() {
        let data = [[1.0_f32, 4.0], [2.0, 8.0], [5.0, 1.0], [9.0, 3.0]];
        let scaled = data.map(|x| x.map(|v| v * 100.0 + 7.0));

        let a = Standardize::fit(&data);
        let b = Standardize::fit(&scaled);
        for (x, y) in data.iter().zip(&scaled) {
            for (u, v) in a.forward(x).iter().zip(b.forward(y)) {
                assert_close(*u, v, 1e-4);
            }
        }
    }

    #[test]
    fn a_constant_input_is_shifted_but_not_scaled() {
        let layer = Standardize::fit(&[[3.0_f32, 1.0], [3.0, 5.0], [3.0, 9.0]]);
        assert_eq!(layer.std_dev()[0], 1.0);
        assert_eq!(layer.forward(&[3.0, 5.0])[0], 0.0);
        assert!(layer.forward(&[4.0, 5.0])[0].is_finite());
    }

    #[test]
    fn fit_works_with_f64_and_iterators() {
        let data = [[1.0_f64], [2.0], [3.0]];
        let layer = Standardize::fit(data.iter());
        assert!((layer.mean()[0] - 2.0).abs() < 1e-12);
        assert!((layer.std_dev()[0] - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn welford_stays_accurate_for_a_huge_offset() {
        // Eine naive Rechnung "Mittel der Quadrate minus Quadrat des Mittels" verliert hier alles.
        let data: Vec<[f64; 1]> = (0..1000).map(|i| [1.0e9 + f64::from(i % 3)]).collect();
        let layer = Standardize::fit(&data);
        let expected = (2.0_f64 / 3.0).sqrt();
        assert!(
            (layer.std_dev()[0] - expected).abs() < 1e-3,
            "{}",
            layer.std_dev()[0]
        );
    }

    #[test]
    #[should_panic(expected = "mindestens ein Beispiel")]
    fn fit_needs_data() {
        let empty: [[f32; 1]; 0] = [];
        let _ = Standardize::fit(&empty);
    }
}
