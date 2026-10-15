//! Weitere Aktivierungen: Werte an bekannten Stellen und Ableitungen gegen die numerische Ableitung.

use neuronws_layers::activation::{Activation, HardSigmoid, HardTanh, LeakyRelu};
use neuronws_layers::gradcheck::assert_activation_matches;

/// Stellen ohne Knick, links und rechts von 0 und weit draußen.
const POINTS: [f64; 9] = [-6.0, -2.5, -1.2, -0.4, 0.3, 0.9, 1.7, 3.4, 8.0];

#[test]
fn leaky_relu_lets_a_little_through_on_the_left() {
    let leaky = LeakyRelu(0.1);
    assert_eq!(leaky.apply(2.0_f32), 2.0);
    assert!((leaky.apply(-2.0_f32) - -0.2).abs() < 1e-7);
    assert_eq!(leaky.derivative(2.0_f32), 1.0);
    assert!((leaky.derivative(-2.0_f32) - 0.1).abs() < 1e-7);
    assert_eq!(LeakyRelu::default().0, 0.01);
    assert_activation_matches(&leaky, &POINTS, 1e-6);
}

#[test]
fn hard_tanh_clamps_and_has_no_slope_outside() {
    assert_eq!(HardTanh.apply(0.25_f32), 0.25);
    assert_eq!(HardTanh.apply(7.0_f32), 1.0);
    assert_eq!(HardTanh.apply(-7.0_f32), -1.0);
    assert_eq!(HardTanh.derivative(0.25_f32), 1.0);
    assert_eq!(HardTanh.derivative(7.0_f32), 0.0);
    assert_eq!(HardTanh.derivative(-7.0_f32), 0.0);
    assert_activation_matches(&HardTanh, &[-3.0, -0.7, 0.2, 0.99, 4.0], 1e-6);
}

#[test]
fn hard_sigmoid_is_a_line_between_minus_three_and_three() {
    assert_eq!(HardSigmoid.apply(0.0_f32), 0.5);
    assert_eq!(HardSigmoid.apply(-3.0_f32), 0.0);
    assert_eq!(HardSigmoid.apply(3.0_f32), 1.0);
    assert_eq!(HardSigmoid.apply(-9.0_f32), 0.0);
    assert_eq!(HardSigmoid.apply(9.0_f32), 1.0);
    assert!((HardSigmoid.derivative(1.0_f32) - 1.0 / 6.0).abs() < 1e-7);
    assert_eq!(HardSigmoid.derivative(5.0_f32), 0.0);
    assert_activation_matches(&HardSigmoid, &[-5.0, -2.0, 0.5, 2.9, 4.0], 1e-6);
}

#[test]
fn the_hard_activations_work_for_f64_too() {
    assert_eq!(HardTanh.apply(0.5_f64), 0.5);
    assert_eq!(HardSigmoid.apply(0.0_f64), 0.5);
    assert_eq!(LeakyRelu(0.5).apply(-2.0_f64), -1.0);
}

#[cfg(any(feature = "std", feature = "libm"))]
mod smooth {
    use super::*;
    use neuronws_layers::activation::{Elu, Gelu, Sigmoid, Silu, Softplus, Swish};

    #[test]
    fn elu_runs_into_minus_alpha_on_the_left() {
        let elu = Elu(2.0);
        assert_eq!(elu.apply(3.0_f64), 3.0);
        assert!((elu.apply(-50.0_f64) - -2.0).abs() < 1e-12);
        assert!((elu.apply(-1.0_f64) - 2.0 * (-1.0_f64).exp_m1()).abs() < 1e-12);
        assert_eq!(Elu::default().0, 1.0);
        assert_activation_matches(&elu, &POINTS, 1e-6);
    }

    #[test]
    fn gelu_matches_known_values() {
        // Werte der Tanh-Näherung (Referenz: PyTorch, approximate = "tanh")
        assert!(Gelu.apply(0.0_f64).abs() < 1e-12);
        assert!((Gelu.apply(1.0_f64) - 0.841_192).abs() < 1e-5);
        assert!((Gelu.apply(-1.0_f64) - -0.158_808).abs() < 1e-5);
        assert!((Gelu.apply(3.0_f64) - 2.996_363).abs() < 1e-5);
        assert_activation_matches(&Gelu, &POINTS, 1e-6);
    }

    #[test]
    fn swish_and_silu() {
        assert!(Silu.apply(0.0_f64).abs() < 1e-12);
        // z · σ(z) bei z = 2
        let sigma = 1.0 / (1.0 + (-2.0_f64).exp());
        assert!((Silu.apply(2.0_f64) - 2.0 * sigma).abs() < 1e-12);
        // Swish mit β = 1 ist Silu, mit anderem β eine andere Kurve
        assert_eq!(Swish(1.0).apply(1.5_f64), Silu.apply(1.5_f64));
        assert_ne!(Swish(2.0).apply(1.5_f64), Silu.apply(1.5_f64));
        assert_activation_matches(&Silu, &POINTS, 1e-6);
        assert_activation_matches(&Swish(2.5), &POINTS, 1e-6);
        // nicht monoton: direkt links von 0 liegt ein flaches Minimum unter 0
        assert!(Silu.apply(-1.28_f64) < Silu.apply(-4.0_f64));
        assert!(Silu.apply(-1.28_f64) < 0.0);
    }

    #[test]
    fn softplus_is_a_smooth_relu() {
        assert!((Softplus.apply(0.0_f64) - core::f64::consts::LN_2).abs() < 1e-12);
        assert!((Softplus.apply(30.0_f64) - 30.0).abs() < 1e-9);
        assert!(Softplus.apply(-30.0_f64) > 0.0);
        assert!(Softplus.apply(-30.0_f64) < 1e-12);
        // Die Ableitung ist Sigmoid.
        for z in POINTS {
            assert!((Softplus.derivative(z) - Sigmoid.apply(z)).abs() < 1e-12);
        }
        assert_activation_matches(&Softplus, &POINTS, 1e-6);
    }

    #[test]
    fn softplus_does_not_overflow_for_large_inputs() {
        assert_eq!(Softplus.apply(1000.0_f32), 1000.0);
        assert_eq!(Softplus.apply(-1000.0_f32), 0.0);
        assert!(Softplus.apply(88.0_f32).is_finite());
    }

    #[test]
    fn the_smooth_activations_also_work_for_f32() {
        for z in [-2.0_f32, -0.5, 0.5, 2.0] {
            for value in [
                Elu(1.0).apply(z),
                Gelu.apply(z),
                Silu.apply(z),
                Softplus.apply(z),
                Elu(1.0).derivative(z),
                Gelu.derivative(z),
                Silu.derivative(z),
                Softplus.derivative(z),
            ] {
                assert!(value.is_finite());
            }
        }
    }

    #[test]
    fn they_work_inside_a_layer() {
        use neuronws_layers::activation_layer::ActivationLayer;
        use neuronws_layers::traits::ForwardPath;

        let layer = ActivationLayer::<3, _>::new(Gelu);
        let [a, b, c] = layer.forward(&[-1.0, 0.0, 1.0]);
        assert!(a < 0.0 && b == 0.0 && c > 0.8);
    }
}
