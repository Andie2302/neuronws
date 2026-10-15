use crate::scalar::Scalar;

/// Eine skalare Aktivierungsfunktion `φ` samt Ableitung `φ'`.
///
/// Beides steht in einem Trait, damit man die Ableitung nicht vergessen kann.
/// Beide Methoden bekommen die Eingabe `z` vor der Aktivierung.
pub trait Activation {
    fn apply<T: Scalar>(&self, z: T) -> T;
    fn derivative<T: Scalar>(&self, z: T) -> T;

    /// Wie `derivative`, bekommt aber zusätzlich das schon bekannte Ergebnis `y = φ(z)`.
    ///
    /// Bei Sigmoid und Tanh lässt sich die Ableitung aus `y` berechnen, ohne `exp` oder `tanh`
    /// noch einmal aufzurufen. Der Standard ignoriert `y`.
    fn derivative_with_output<T: Scalar>(&self, z: T, _y: T) -> T {
        self.derivative(z)
    }
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
        sigmoid(z)
    }

    /// φ'(z) = φ(z) · (1 − φ(z))
    fn derivative<T: Scalar>(&self, z: T) -> T {
        self.derivative_with_output(z, self.apply(z))
    }

    fn derivative_with_output<T: Scalar>(&self, _z: T, y: T) -> T {
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
        self.derivative_with_output(z, self.apply(z))
    }

    fn derivative_with_output<T: Scalar>(&self, _z: T, y: T) -> T {
        T::ONE - y * y
    }
}

/// Leaky ReLU: φ(z) = z für z > 0, sonst α·z.
///
/// Wie ReLU, lässt aber links von 0 ein kleines Stück Signal durch (Standard α = 0,01). So
/// bekommt ein Neuron, das einmal nur negative Werte sah, weiter einen Gradienten und kann sich
/// erholen ("sterbendes ReLU"). Braucht keine Mathe-Funktionen.
#[derive(Debug, Clone, Copy)]
pub struct LeakyRelu(pub f32);

impl Default for LeakyRelu {
    fn default() -> Self {
        Self(0.01)
    }
}

impl Activation for LeakyRelu {
    fn apply<T: Scalar>(&self, z: T) -> T {
        if z > T::ZERO {
            z
        } else {
            T::from_f32(self.0) * z
        }
    }

    fn derivative<T: Scalar>(&self, z: T) -> T {
        if z > T::ZERO {
            T::ONE
        } else {
            T::from_f32(self.0)
        }
    }
}

/// Hartes Tanh: φ(z) = z, begrenzt auf [−1, 1].
///
/// Die Form von Tanh ohne `exp`, deshalb auch ohne `std` und `libm` nutzbar. Außerhalb von
/// (−1, 1) ist die Ableitung 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct HardTanh;

impl Activation for HardTanh {
    fn apply<T: Scalar>(&self, z: T) -> T {
        z.clamp(-T::ONE, T::ONE)
    }

    fn derivative<T: Scalar>(&self, z: T) -> T {
        if z > -T::ONE && z < T::ONE {
            T::ONE
        } else {
            T::ZERO
        }
    }
}

/// Hartes Sigmoid: φ(z) = z/6 + 1/2, begrenzt auf [0, 1].
///
/// Eine Gerade statt der S-Kurve, ohne `exp` und damit auch ohne `std` und `libm` nutzbar. Es
/// geht bei z = −3 von 0 auf 1 bei z = 3 (die Form aus PyTorch). Außerhalb ist die Ableitung 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct HardSigmoid;

impl Activation for HardSigmoid {
    fn apply<T: Scalar>(&self, z: T) -> T {
        (z / T::from_f32(6.0) + T::from_f32(0.5)).clamp(T::ZERO, T::ONE)
    }

    fn derivative<T: Scalar>(&self, z: T) -> T {
        let three = T::from_f32(3.0);
        if z > -three && z < three {
            T::ONE / T::from_f32(6.0)
        } else {
            T::ZERO
        }
    }
}

/// `1 / (1 + e^(−z))`, auch von mehreren Aktivierungen unten gebraucht.
#[cfg(any(feature = "std", feature = "libm"))]
fn sigmoid<T: Scalar>(z: T) -> T {
    T::ONE / (T::ONE + (-z).exp())
}

/// ELU: φ(z) = z für z > 0, sonst α (e^z − 1).
///
/// Links von 0 läuft die Kurve glatt auf −α zu (Standard α = 1), statt wie bei ReLU abrupt bei 0
/// zu enden. Das hält den Mittelwert der Ausgaben näher an 0. Braucht `exp`, deshalb nur mit dem
/// Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy)]
pub struct Elu(pub f32);

#[cfg(any(feature = "std", feature = "libm"))]
impl Default for Elu {
    fn default() -> Self {
        Self(1.0)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl Activation for Elu {
    fn apply<T: Scalar>(&self, z: T) -> T {
        if z > T::ZERO {
            z
        } else {
            T::from_f32(self.0) * (z.exp() - T::ONE)
        }
    }

    /// Links von 0: φ'(z) = α e^z = φ(z) + α
    fn derivative<T: Scalar>(&self, z: T) -> T {
        if z > T::ZERO {
            T::ONE
        } else {
            T::from_f32(self.0) * z.exp()
        }
    }

    fn derivative_with_output<T: Scalar>(&self, z: T, y: T) -> T {
        if z > T::ZERO {
            T::ONE
        } else {
            y + T::from_f32(self.0)
        }
    }
}

/// GELU (Gaussian Error Linear Unit), in der üblichen Näherung mit Tanh:
/// φ(z) = ½ z (1 + tanh(√(2/π) (z + 0,044715 z³))).
///
/// Eine glatte Verwandte von ReLU, die in Transformer-Netzen Standard ist. Braucht `tanh`,
/// deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Gelu;

#[cfg(any(feature = "std", feature = "libm"))]
impl Gelu {
    const CUBIC: f64 = 0.044_715;
    /// √(2/π)
    const SCALE: f64 = 0.797_884_560_802_865_4;

    fn tanh_part<T: Scalar>(z: T) -> T {
        let inner = T::from_f64(Self::SCALE) * (z + T::from_f64(Self::CUBIC) * z * z * z);
        inner.tanh()
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl Activation for Gelu {
    fn apply<T: Scalar>(&self, z: T) -> T {
        T::from_f32(0.5) * z * (T::ONE + Self::tanh_part(z))
    }

    /// φ'(z) = ½ (1 + t) + ½ z (1 − t²) √(2/π) (1 + 3 · 0,044715 z²) mit t = tanh(…)
    fn derivative<T: Scalar>(&self, z: T) -> T {
        let half = T::from_f32(0.5);
        let t = Self::tanh_part(z);
        let inner_slope =
            T::from_f64(Self::SCALE) * (T::ONE + T::from_f64(3.0 * Self::CUBIC) * z * z);
        half * (T::ONE + t) + half * z * (T::ONE - t * t) * inner_slope
    }
}

/// Swish: φ(z) = z · σ(β z) mit der Sigmoid-Kurve σ. Bei β = 1 heißt es [`Silu`].
///
/// Glatt, unten begrenzt und nicht monoton: Direkt links von 0 sinkt die Kurve leicht unter 0,
/// bevor sie sich dem Wert 0 nähert. Braucht `exp`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy)]
pub struct Swish(pub f32);

#[cfg(any(feature = "std", feature = "libm"))]
impl Default for Swish {
    fn default() -> Self {
        Self(1.0)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl Activation for Swish {
    fn apply<T: Scalar>(&self, z: T) -> T {
        z * sigmoid(T::from_f32(self.0) * z)
    }

    /// φ'(z) = σ + β z σ (1 − σ) mit σ = σ(β z)
    fn derivative<T: Scalar>(&self, z: T) -> T {
        let beta = T::from_f32(self.0);
        let s = sigmoid(beta * z);
        s + beta * z * s * (T::ONE - s)
    }
}

/// SiLU (Sigmoid Linear Unit): φ(z) = z · σ(z), also [`Swish`] mit β = 1.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Silu;

#[cfg(any(feature = "std", feature = "libm"))]
impl Activation for Silu {
    fn apply<T: Scalar>(&self, z: T) -> T {
        Swish(1.0).apply(z)
    }

    fn derivative<T: Scalar>(&self, z: T) -> T {
        Swish(1.0).derivative(z)
    }
}

/// Softplus: φ(z) = ln(1 + e^z), eine glatte Version von ReLU.
///
/// Die Ableitung ist die Sigmoid-Kurve. Gerechnet wird als `max(z, 0) + ln(1 + e^(−|z|))`, damit
/// `e^z` bei großem `z` nicht überläuft. Braucht `exp` und `ln`, deshalb nur mit dem Feature
/// `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Softplus;

#[cfg(any(feature = "std", feature = "libm"))]
impl Activation for Softplus {
    fn apply<T: Scalar>(&self, z: T) -> T {
        z.max(T::ZERO) + (T::ONE + (-z.abs()).exp()).ln()
    }

    /// φ'(z) = σ(z)
    fn derivative<T: Scalar>(&self, z: T) -> T {
        sigmoid(z)
    }
}
