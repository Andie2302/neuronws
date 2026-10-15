//! Die kleinen Zahlen-Helfer von `Scalar` brauchen weder `std` noch `libm`.

use neuronws_layers::scalar::Scalar;

fn comparisons_work<T: Scalar>() {
    let (one, two) = (T::ONE, T::TWO);
    let (minus_one, minus_two) = (-one, -two);

    assert!(minus_two.abs() == two);
    assert!(two.abs() == two);
    assert!(T::ZERO.abs() == T::ZERO);

    assert!(two.sign() == one);
    assert!(minus_two.sign() == minus_one);
    assert!(T::ZERO.sign() == T::ZERO);

    assert!(one.min(two) == one);
    assert!(two.min(one) == one);
    assert!(one.max(two) == two);
    assert!(two.max(one) == two);

    let low = minus_one;
    let high = one;
    assert!(T::from_f32(5.0).clamp(low, high) == high);
    assert!(T::from_f32(-5.0).clamp(low, high) == low);
    assert!(T::from_f32(0.25).clamp(low, high) == T::from_f32(0.25));
}

#[test]
fn abs_sign_min_max_and_clamp_for_f32_and_f64() {
    comparisons_work::<f32>();
    comparisons_work::<f64>();
}

#[test]
fn nan_is_passed_through_not_hidden() {
    // Ein Wert, der keine Zahl ist, soll nicht stillschweigend zu 0 oder 1 werden.
    assert!(Scalar::abs(f32::NAN).is_nan());
    assert_eq!(Scalar::sign(f32::NAN), 0.0);
    assert!(Scalar::clamp(f32::NAN, -1.0, 1.0).is_nan());
}

#[test]
fn from_f64_rounds_for_f32_and_is_exact_for_f64() {
    assert_eq!(<f32 as Scalar>::from_f64(0.5), 0.5_f32);
    assert_eq!(<f64 as Scalar>::from_f64(0.1), 0.1_f64);
    assert_eq!(<f32 as Scalar>::from_f64(0.1), 0.1_f32);
}

fn round_trip<T: Scalar>(value: T) {
    let mut buffer = [0_u8; 16];
    value.write_le(&mut buffer);
    assert!(T::read_le(&buffer).to_f64() == value.to_f64());
    // Der Rest des Puffers bleibt unberührt.
    assert!(buffer[T::BYTES..].iter().all(|&b| b == 0));
}

#[test]
fn bytes_are_little_endian_and_round_trip() {
    assert_eq!(<f32 as Scalar>::BYTES, 4);
    assert_eq!(<f64 as Scalar>::BYTES, 8);

    let mut buffer = [0_u8; 8];
    1.0_f32.write_le(&mut buffer);
    assert_eq!(&buffer[..4], &[0x00, 0x00, 0x80, 0x3f]);

    for value in [
        0.0_f32,
        -1.5,
        core::f32::consts::PI,
        f32::MAX,
        f32::MIN_POSITIVE,
    ] {
        round_trip(value);
    }
    for value in [
        0.0_f64,
        -1.5,
        core::f64::consts::PI,
        f64::MAX,
        f64::MIN_POSITIVE,
    ] {
        round_trip(value);
    }
}

#[test]
#[should_panic]
fn writing_into_a_too_short_buffer_panics() {
    1.0_f64.write_le(&mut [0_u8; 7]);
}

#[test]
#[should_panic]
fn reading_from_a_too_short_slice_panics() {
    let _ = <f32 as Scalar>::read_le(&[0_u8; 3]);
}

#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn cos_and_sin_agree_with_the_identity() {
    fn check<T: Scalar>(tolerance: f64) {
        for x in [-3.0_f32, -1.0, 0.0, 0.5, 2.0, 7.0] {
            let x = T::from_f32(x);
            let one = x.cos() * x.cos() + x.sin() * x.sin();
            assert!((one.to_f64() - 1.0).abs() < tolerance);
        }
        assert!((T::ZERO.cos().to_f64() - 1.0).abs() < tolerance);
        assert!(T::ZERO.sin().to_f64().abs() < tolerance);
    }
    check::<f32>(1e-6);
    check::<f64>(1e-12);
}
