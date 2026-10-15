/// Eine skalare Aktivierungsfunktion `φ` samt Ableitung `φ'`.
///
/// Beides steht in einem Trait, damit man die Ableitung nicht vergessen kann.
/// Beide Methoden bekommen die Eingabe `z` vor der Aktivierung.
pub trait Activation {
    fn apply(&self, z: f32) -> f32;
    fn derivative(&self, z: f32) -> f32;
}

/// φ(z) = z
pub struct Identity;

impl Activation for Identity {
    fn apply(&self, z: f32) -> f32 {
        z
    }

    fn derivative(&self, _z: f32) -> f32 {
        1.0
    }
}

/// Schwellenwert (Heaviside): 1 für z ≥ 0, sonst 0. Die Aktivierung des klassischen Perceptrons.
///
/// Die Ableitung ist überall 0 (bei 0 nicht definiert), deshalb eignet sich `Step` nicht für
/// gradientenbasiertes Lernen. Das Perceptron lernt mit seiner eigenen Regel.
pub struct Step;

impl Activation for Step {
    fn apply(&self, z: f32) -> f32 {
        if z >= 0.0 { 1.0 } else { 0.0 }
    }

    fn derivative(&self, _z: f32) -> f32 {
        0.0
    }
}

/// φ(z) = max(0, z)
pub struct Relu;

impl Activation for Relu {
    fn apply(&self, z: f32) -> f32 {
        z.max(0.0)
    }

    fn derivative(&self, z: f32) -> f32 {
        if z > 0.0 { 1.0 } else { 0.0 }
    }
}

/// φ(z) = 1 / (1 + e^(−z)), drückt alle Werte in den Bereich (0, 1).
///
/// Braucht `exp`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
pub struct Sigmoid;

#[cfg(any(feature = "std", feature = "libm"))]
impl Activation for Sigmoid {
    fn apply(&self, z: f32) -> f32 {
        1.0 / (1.0 + crate::math::exp(-z))
    }

    /// φ'(z) = φ(z) · (1 − φ(z))
    fn derivative(&self, z: f32) -> f32 {
        let y = self.apply(z);
        y * (1.0 - y)
    }
}

/// φ(z) = tanh(z), drückt alle Werte in den Bereich (−1, 1).
///
/// Braucht `tanh`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
pub struct Tanh;

#[cfg(any(feature = "std", feature = "libm"))]
impl Activation for Tanh {
    fn apply(&self, z: f32) -> f32 {
        crate::math::tanh(z)
    }

    /// φ'(z) = 1 − tanh²(z)
    fn derivative(&self, z: f32) -> f32 {
        let y = self.apply(z);
        1.0 - y * y
    }
}
