use neuronws_layers::loss::{Loss, MeanSquaredError};

#[test]
fn mse_value_is_mean_of_squared_errors() {
    let loss = MeanSquaredError.value(&[1.0, 3.0], &[0.0, 1.0]);
    assert_eq!(loss, (1.0 + 4.0) / 2.0);
}

#[test]
fn mse_is_zero_for_a_perfect_prediction() {
    assert_eq!(
        MeanSquaredError.value(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
        0.0
    );
}

#[test]
fn mse_gradient_matches_numeric_gradient() {
    const Y: [f32; 3] = [0.5, -1.0, 2.0];
    const T: [f32; 3] = [1.0, 0.0, 1.5];
    const EPS: f32 = 1e-2;

    let analytic = MeanSquaredError.gradient(&Y, &T);
    for (i, a) in analytic.iter().enumerate() {
        let mut up = Y;
        up[i] += EPS;
        let mut down = Y;
        down[i] -= EPS;
        let numeric =
            (MeanSquaredError.value(&up, &T) - MeanSquaredError.value(&down, &T)) / (2.0 * EPS);
        assert!((a - numeric).abs() < 1e-3, "i = {i}: {a} gegen {numeric}");
    }
}
