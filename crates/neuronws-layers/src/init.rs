use crate::random::Rng;
use crate::scalar::Scalar;

/// Wie die Startwerte der Gewichte gewählt werden.
///
/// `fan_in` ist die Zahl der Eingänge eines Neurons, `fan_out` die Zahl der Neuronen der Schicht.
/// Manche Verfahren skalieren die Werte damit, damit `z` am Anfang weder winzig noch riesig ist.
pub trait Initializer {
    fn sample<T: Scalar>(&self, rng: &mut Rng, fan_in: usize, fan_out: usize) -> T;
}

/// Gleichverteilt in `[-1, 1)`, dann mit `limit` multipliziert.
fn uniform<T: Scalar>(rng: &mut Rng, limit: T) -> T {
    limit * (T::TWO * T::from_f32(rng.next_f32()) - T::ONE)
}

/// Alle Werte 0. Für Gewichte ungeeignet (alle Neuronen würden dasselbe lernen), aber nützlich
/// für Tests und für Bias-Werte.
#[derive(Debug, Clone, Copy, Default)]
pub struct Zeros;

impl Initializer for Zeros {
    fn sample<T: Scalar>(&self, _rng: &mut Rng, _fan_in: usize, _fan_out: usize) -> T {
        T::ZERO
    }
}

/// Immer derselbe Wert.
#[derive(Debug, Clone, Copy)]
pub struct Constant(pub f32);

impl Initializer for Constant {
    fn sample<T: Scalar>(&self, _rng: &mut Rng, _fan_in: usize, _fan_out: usize) -> T {
        T::from_f32(self.0)
    }
}

/// Gleichverteilt in `[-limit, limit)`.
#[derive(Debug, Clone, Copy)]
pub struct Uniform(pub f32);

impl Initializer for Uniform {
    fn sample<T: Scalar>(&self, rng: &mut Rng, _fan_in: usize, _fan_out: usize) -> T {
        uniform(rng, T::from_f32(self.0))
    }
}

/// Xavier (Glorot): gleichverteilt in `±√(6 / (fan_in + fan_out))`. Passt zu Tanh und Sigmoid.
///
/// Braucht die Wurzel, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Xavier;

#[cfg(any(feature = "std", feature = "libm"))]
impl Initializer for Xavier {
    fn sample<T: Scalar>(&self, rng: &mut Rng, fan_in: usize, fan_out: usize) -> T {
        let limit = (T::from_f32(6.0) / T::from_usize(fan_in + fan_out)).sqrt();
        uniform(rng, limit)
    }
}

/// He: gleichverteilt in `±√(6 / fan_in)`. Passt zu ReLU, weil die Hälfte der Werte wegfällt.
///
/// Braucht die Wurzel, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct He;

#[cfg(any(feature = "std", feature = "libm"))]
impl Initializer for He {
    fn sample<T: Scalar>(&self, rng: &mut Rng, fan_in: usize, _fan_out: usize) -> T {
        let limit = (T::from_f32(6.0) / T::from_usize(fan_in)).sqrt();
        uniform(rng, limit)
    }
}
