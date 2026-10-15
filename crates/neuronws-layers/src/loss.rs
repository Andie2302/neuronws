use crate::scalar::Scalar;

/// Eine Verlustfunktion: misst, wie weit die Ausgabe vom Ziel entfernt ist.
///
/// `gradient` ist `∂L/∂y` und damit der Startpunkt des Backward-Passes.
pub trait Loss<const N: usize> {
    fn value<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> T;
    fn gradient<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> [T; N];
}

/// Mittlerer quadratischer Fehler: `L = (1/N) Σ (y − t)²`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanSquaredError;

impl<const N: usize> Loss<N> for MeanSquaredError {
    fn value<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> T {
        let sum = output
            .iter()
            .zip(target)
            .fold(T::ZERO, |sum, (&y, &t)| sum + (y - t) * (y - t));
        sum / T::from_usize(N)
    }

    /// `∂L/∂y[i] = 2 (y[i] − t[i]) / N`
    fn gradient<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> [T; N] {
        core::array::from_fn(|i| T::TWO * (output[i] - target[i]) / T::from_usize(N))
    }
}

/// Binäre Kreuzentropie: `L = −(1/N) Σ [t ln(y) + (1 − t) ln(1 − y)]`.
///
/// Für Ausgaben `y` in (0, 1), also hinter einer Sigmoid-Schicht, und Ziele `t` von 0 oder 1.
/// `y` wird knapp vor 0 und 1 abgeschnitten, damit `ln(0)` nie auftritt.
/// Braucht `ln`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryCrossEntropy;

#[cfg(any(feature = "std", feature = "libm"))]
fn clamp_open_unit<T: Scalar>(y: T) -> T {
    let epsilon = T::from_f32(1e-7);
    if y < epsilon {
        epsilon
    } else if y > T::ONE - epsilon {
        T::ONE - epsilon
    } else {
        y
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<const N: usize> Loss<N> for BinaryCrossEntropy {
    fn value<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> T {
        let sum = output.iter().zip(target).fold(T::ZERO, |sum, (&y, &t)| {
            let y = clamp_open_unit(y);
            sum - (t * y.ln() + (T::ONE - t) * (T::ONE - y).ln())
        });
        sum / T::from_usize(N)
    }

    /// `∂L/∂y[i] = (y[i] − t[i]) / (y[i] (1 − y[i]) N)`
    fn gradient<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> [T; N] {
        core::array::from_fn(|i| {
            let y = clamp_open_unit(output[i]);
            (y - target[i]) / (y * (T::ONE - y) * T::from_usize(N))
        })
    }
}
