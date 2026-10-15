//! Welche Quelle liefert `exp`, `tanh`, `sqrt` und `ln`: `std` oder `libm`?
//!
//! `libm` hat Vorrang, wenn beide Features an sind. Das prüft dieser Test mit dem Bitmuster der
//! Ergebnisse. Er läuft in jeder Feature-Kombination und prüft die jeweils richtige Quelle.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::scalar::Scalar;

/// Eine Auswahl von Eingaben, gleichmäßig über einen großen Bereich verteilt.
fn samples() -> impl Iterator<Item = f32> {
    (-4000..4000).map(|i| i as f32 * 0.0123 + 0.0007)
}

#[cfg(feature = "libm")]
#[test]
fn libm_is_used_when_the_feature_is_on() {
    for x in samples() {
        assert_eq!(
            Scalar::exp(x).to_bits(),
            libm::expf(x).to_bits(),
            "exp({x})"
        );
        assert_eq!(
            Scalar::tanh(x).to_bits(),
            libm::tanhf(x).to_bits(),
            "tanh({x})"
        );
        assert_eq!(
            Scalar::sqrt(x.abs()).to_bits(),
            libm::sqrtf(x.abs()).to_bits()
        );
        assert_eq!(
            Scalar::ln(x.abs() + 0.5).to_bits(),
            libm::logf(x.abs() + 0.5).to_bits()
        );
    }
    for x in samples().map(f64::from) {
        assert_eq!(Scalar::exp(x).to_bits(), libm::exp(x).to_bits(), "exp({x})");
        assert_eq!(
            Scalar::tanh(x).to_bits(),
            libm::tanh(x).to_bits(),
            "tanh({x})"
        );
    }
}

#[cfg(all(feature = "std", not(feature = "libm")))]
#[test]
fn std_is_used_when_only_std_is_on() {
    for x in samples() {
        assert_eq!(Scalar::exp(x).to_bits(), x.exp().to_bits(), "exp({x})");
        assert_eq!(Scalar::tanh(x).to_bits(), x.tanh().to_bits(), "tanh({x})");
        assert_eq!(Scalar::sqrt(x.abs()).to_bits(), x.abs().sqrt().to_bits());
        assert_eq!(
            Scalar::ln(x.abs() + 0.5).to_bits(),
            (x.abs() + 0.5).ln().to_bits()
        );
    }
}

/// Der Test ist nur aussagekräftig, wenn `std` und `libm` auf dieser Plattform überhaupt
/// verschiedene Bitmuster liefern. Sonst könnte man sie nicht unterscheiden.
#[cfg(all(feature = "std", feature = "libm"))]
#[test]
fn std_and_libm_really_differ_somewhere_so_the_precedence_is_observable() {
    let differing = samples()
        .filter(|&x| x.exp().to_bits() != libm::expf(x).to_bits())
        .count()
        + samples()
            .filter(|&x| x.tanh().to_bits() != libm::tanhf(x).to_bits())
            .count();
    assert!(differing > 0, "std und libm stimmen hier überall überein");
}

#[test]
fn results_are_close_to_the_true_values_either_way() {
    for x in [-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
        assert!((Scalar::exp(x) - x.exp()).abs() <= 1e-5 * x.exp().max(1.0));
        assert!((Scalar::tanh(x) - x.tanh()).abs() < 1e-6);
    }
    assert!((Scalar::sqrt(2.0_f32) - core::f32::consts::SQRT_2).abs() < 1e-6);
    assert!((Scalar::ln(core::f32::consts::E) - 1.0).abs() < 1e-6);
}
