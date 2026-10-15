use crate::scalar::Scalar;

/// Eine skalare Aktivierungsfunktion `φ` samt Ableitung `φ'`.
///
/// Beides steht in einem Trait, damit man die Ableitung nicht vergessen kann.
/// Beide Methoden bekommen die Eingabe `z` vor der Aktivierung.
pub trait Activation {
    fn apply<T: Scalar>(&self, z: T) -> T;
    fn derivative<T: Scalar>(&self, z: T) -> T;
}

/// φ(z) = z
#[derive(Debug, Clone, Copy, Default)]
pub struct Identity;

impl Activation for Identity {
    fn apply<T: Scalar>(&self, z: T) -> T {
        z
    }

    fn derivative<T: Scalar>(&self, _z: T) -> T {
        T::ONE
    }
}

/// Schwellenwert (Heaviside): 1 für z ≥ 0, sonst 0. Die Aktivierung des klassischen Perceptrons.
///
/// Die Ableitung ist überall 0 (bei 0 nicht definiert), deshalb eignet sich `Step` nicht für
/// gradientenbasiertes Lernen. Das Perceptron lernt mit seiner eigenen Regel.
#[derive(Debug, Clone, Copy, Default)]
pub struct Step;

impl Activation for Step {
    fn apply<T: Scalar>(&self, z: T) -> T {
        if z >= T::ZERO { T::ONE } else { T::ZERO }
    }

    fn derivative<T: Scalar>(&self, _z: T) -> T {
        T::ZERO
    }
}

/// φ(z) = max(0, z)
#[derive(Debug, Clone, Copy, Default)]
pub struct Relu;

impl Activation for Relu {
    fn apply<T: Scalar>(&self, z: T) -> T {
        if z > T::ZERO { z } else { T::ZERO }
    }

    fn derivative<T: Scalar>(&self, z: T) -> T {
        if z > T::ZERO { T::ONE } else { T::ZERO }
    }
}

/// φ(z) = 1 / (1 + e^(−z)), drückt alle Werte in den Bereich (0, 1).
///
/// Braucht `exp`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;

#[cfg(any(feature = "std", feature = "libm"))]
impl Activation for Sigmoid {
    fn apply<T: Scalar>(&self, z: T) -> T {
        T::ONE / (T::ONE + (-z).exp())
    }

    /// φ'(z) = φ(z) · (1 − φ(z))
    fn derivative<T: Scalar>(&self, z: T) -> T {
        let y = self.apply(z);
        y * (T::ONE - y)
    }
}

/// φ(z) = tanh(z), drückt alle Werte in den Bereich (−1, 1).
///
/// Braucht `tanh`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh;

#[cfg(any(feature = "std", feature = "libm"))]
impl Activation for Tanh {
    fn apply<T: Scalar>(&self, z: T) -> T {
        z.tanh()
    }

    /// φ'(z) = 1 − tanh²(z)
    fn derivative<T: Scalar>(&self, z: T) -> T {
        let y = self.apply(z);
        T::ONE - y * y
    }
}
