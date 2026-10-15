use neuronws_layers::dense_layer::{DenseLayer, Linear};
use neuronws_layers::init::{Constant, Initializer, Uniform, Zeros};
use neuronws_layers::random::Rng;

fn all_weights<const I: usize, const O: usize>(layer: &DenseLayer<I, O>) -> Vec<f32> {
    layer.weights().iter().flatten().copied().collect()
}

#[test]
fn zeros_gives_zero_weights_and_zero_bias() {
    let layer = DenseLayer::<3, 2>::initialized(Linear::new(), &Zeros, &mut Rng::new(1));
    assert!(all_weights(&layer).iter().all(|&w| w == 0.0));
    assert_eq!(layer.bias(), &[0.0; 2]);
}

#[test]
fn constant_gives_the_same_weight_everywhere_but_bias_stays_zero() {
    let layer = DenseLayer::<3, 2>::initialized(Linear::new(), &Constant(0.5), &mut Rng::new(1));
    assert!(all_weights(&layer).iter().all(|&w| w == 0.5));
    assert_eq!(layer.bias(), &[0.0; 2]);
}

#[test]
fn uniform_stays_within_its_limit_and_varies() {
    let layer = DenseLayer::<4, 4>::initialized(Linear::new(), &Uniform(0.3), &mut Rng::new(2));
    let weights = all_weights(&layer);
    assert!(weights.iter().all(|w| (-0.3..0.3).contains(w)));
    assert!(weights.iter().any(|&w| w != weights[0]));
    assert!(weights.iter().any(|&w| w < 0.0) && weights.iter().any(|&w| w > 0.0));
}

#[test]
fn the_same_seed_gives_the_same_layer() {
    let make =
        |seed| DenseLayer::<3, 3>::initialized(Linear::new(), &Uniform(1.0), &mut Rng::new(seed));
    assert_eq!(make(5).weights(), make(5).weights());
    assert_ne!(make(5).weights(), make(6).weights());
}

#[test]
fn uniform_matches_the_older_random_constructor() {
    let a = DenseLayer::<2, 3>::random(&mut Rng::new(9), 0.5);
    let b = DenseLayer::<2, 3>::initialized(Linear::new(), &Uniform(0.5), &mut Rng::new(9));
    assert_eq!(a.weights(), b.weights());
}

#[test]
fn an_initializer_can_depend_on_the_layer_size() {
    /// Teilt durch die Anzahl der Eingänge, damit die Summe vieler Eingänge klein bleibt.
    struct ScaledByFanIn;
    impl Initializer for ScaledByFanIn {
        fn sample<T: neuronws_layers::scalar::Scalar>(
            &self,
            _: &mut Rng,
            fan_in: usize,
            _: usize,
        ) -> T {
            T::ONE / T::from_usize(fan_in)
        }
    }
    let narrow = DenseLayer::<2, 1>::initialized(Linear::new(), &ScaledByFanIn, &mut Rng::new(1));
    let wide = DenseLayer::<10, 1>::initialized(Linear::new(), &ScaledByFanIn, &mut Rng::new(1));
    assert_eq!(narrow.weights(), &[[0.5, 0.5]]);
    assert!(wide.weights()[0].iter().all(|&w| (w - 0.1).abs() < 1e-7));
}

#[test]
fn it_works_for_f64() {
    let layer =
        DenseLayer::<2, 2, f64>::initialized(Linear::new(), &Constant(0.25), &mut Rng::new(1));
    assert_eq!(layer.weights(), &[[0.25; 2]; 2]);
}

#[cfg(any(feature = "std", feature = "libm"))]
mod scaled {
    use neuronws_layers::init::{He, Xavier};

    use super::*;

    #[test]
    fn xavier_limit_depends_on_fan_in_and_fan_out() {
        // 3 + 3 = 6: Grenze 1
        let layer = DenseLayer::<3, 3>::initialized(Linear::new(), &Xavier, &mut Rng::new(4));
        let weights = all_weights(&layer);
        assert!(weights.iter().all(|w| w.abs() <= 1.0));
        assert!(weights.iter().any(|w| w.abs() > 0.5));

        // 150 + 150 = 300: Grenze sqrt(6/300) ≈ 0,14
        let big = DenseLayer::<150, 2>::initialized(Linear::new(), &Xavier, &mut Rng::new(4));
        let limit = (6.0_f32 / 152.0).sqrt();
        assert!(all_weights(&big).iter().all(|w| w.abs() <= limit));
    }

    #[test]
    fn he_limit_depends_on_fan_in_only() {
        let a = DenseLayer::<6, 1>::initialized(Linear::new(), &He, &mut Rng::new(4));
        let b = DenseLayer::<6, 50>::initialized(Linear::new(), &He, &mut Rng::new(4));
        // sqrt(6/6) = 1 in beiden Fällen
        assert!(all_weights(&a).iter().all(|w| w.abs() <= 1.0));
        assert!(all_weights(&b).iter().all(|w| w.abs() <= 1.0));
        assert!(all_weights(&b).iter().any(|w| w.abs() > 0.5));
    }

    #[test]
    fn xavier_matches_the_older_constructor() {
        let a = DenseLayer::<4, 3>::xavier(&mut Rng::new(7));
        let b = DenseLayer::<4, 3>::initialized(Linear::new(), &Xavier, &mut Rng::new(7));
        assert_eq!(a.weights(), b.weights());
    }
}
