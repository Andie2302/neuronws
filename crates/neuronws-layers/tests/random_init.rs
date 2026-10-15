use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::random::Rng;

#[test]
fn random_weights_are_within_the_limit_and_bias_is_zero() {
    let layer = DenseLayer::<3, 4>::random(&mut Rng::new(1), 0.5);
    for w in layer.weights().iter().flatten() {
        assert!((-0.5..=0.5).contains(w), "{w}");
    }
    assert_eq!(layer.bias(), &[0.0; 4]);
}

#[test]
fn random_weights_are_not_all_equal() {
    let layer = DenseLayer::<3, 4>::random(&mut Rng::new(1), 1.0);
    let first = layer.weights()[0][0];
    assert!(layer.weights().iter().flatten().any(|&w| w != first));
}

#[test]
fn same_seed_builds_the_same_layer() {
    let a = DenseLayer::<2, 3>::random(&mut Rng::new(9), 1.0);
    let b = DenseLayer::<2, 3>::random(&mut Rng::new(9), 1.0);
    assert_eq!(a.weights(), b.weights());
}

#[test]
fn consecutive_layers_from_one_rng_differ() {
    let mut rng = Rng::new(9);
    let a = DenseLayer::<2, 3>::random(&mut rng, 1.0);
    let b = DenseLayer::<2, 3>::random(&mut rng, 1.0);
    assert_ne!(a.weights(), b.weights());
}

#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn xavier_limit_is_sqrt_of_six_over_fan_in_plus_fan_out() {
    // IN + OUT = 6 -> Grenze = 1
    let layer = DenseLayer::<2, 4>::xavier(&mut Rng::new(5));
    assert!(layer.weights().iter().flatten().all(|w| w.abs() <= 1.0));
    assert!(layer.weights().iter().flatten().any(|w| w.abs() > 0.5));
}
