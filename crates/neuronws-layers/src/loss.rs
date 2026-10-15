use crate::scalar::Scalar;
use crate::values::Values;

/// Eine Verlustfunktion: misst, wie weit die Ausgabe vom Ziel entfernt ist.
///
/// Ausgabe und Ziel dürfen jede Form haben (Vektor, Matrix, Bild, ...), solange beide gleich sind.
/// Die Verlustfunktion sieht sie als flache Liste von `V::LEN` Zahlen.
///
/// `gradient` ist `∂L/∂y` und damit der Startpunkt des Backward-Passes.
pub trait Loss {
    fn value<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> T;
    fn gradient<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> V;
}

/// Bildet aus jedem Paar `(y[i], t[i])` eine Zahl, in der Reihenfolge von `Values`.
fn map_pairs<T: Scalar, V: Values<T>>(output: &V, target: &V, mut f: impl FnMut(T, T) -> T) -> V {
    let mut result = V::zeros();
    for i in 0..V::LEN {
        *result.value_mut(i) = f(output.value(i), target.value(i));
    }
    result
}

/// Summe über alle Paare `(y[i], t[i])`.
fn sum_pairs<T: Scalar, V: Values<T>>(output: &V, target: &V, mut f: impl FnMut(T, T) -> T) -> T {
    (0..V::LEN).fold(T::ZERO, |sum, i| sum + f(output.value(i), target.value(i)))
}

/// Mittlerer quadratischer Fehler: `L = (1/N) Σ (y − t)²`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn value<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> T {
        sum_pairs(output, target, |y, t| (y - t) * (y - t)) / T::from_usize(V::LEN)
    }

    /// `∂L/∂y[i] = 2 (y[i] − t[i]) / N`
    fn gradient<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> V {
        let n = T::from_usize(V::LEN);
        map_pairs(output, target, |y, t| T::TWO * (y - t) / n)
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
impl Loss for BinaryCrossEntropy {
    fn value<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> T {
        let sum = sum_pairs(output, target, |y, t| {
            let y = clamp_open_unit(y);
            -(t * y.ln() + (T::ONE - t) * (T::ONE - y).ln())
        });
        sum / T::from_usize(V::LEN)
    }

    /// `∂L/∂y[i] = (y[i] − t[i]) / (y[i] (1 − y[i]) N)`
    fn gradient<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> V {
        let n = T::from_usize(V::LEN);
        map_pairs(output, target, |y, t| {
            let y = clamp_open_unit(y);
            (y - t) / (y * (T::ONE - y) * n)
        })
    }
}

/// Kreuzentropie für mehrere Klassen: `L = −Σ t[i] ln(y[i])`.
///
/// Für Ausgaben `y` hinter einer `SoftmaxLayer` und ein Ziel `t` als "One-Hot"-Vektor
/// (eine 1 bei der richtigen Klasse, sonst 0). Anders als beim Mittelwert über `N` Werte
/// wird hier **summiert**: Pro Beispiel zählt nur die richtige Klasse.
/// `y` wird knapp über 0 abgeschnitten, damit `ln(0)` nie auftritt.
/// Braucht `ln`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CategoricalCrossEntropy;

#[cfg(any(feature = "std", feature = "libm"))]
fn clamp_positive<T: Scalar>(y: T) -> T {
    let epsilon = T::from_f32(1e-7);
    if y < epsilon { epsilon } else { y }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl Loss for CategoricalCrossEntropy {
    fn value<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> T {
        sum_pairs(output, target, |y, t| -(t * clamp_positive(y).ln()))
    }

    /// `∂L/∂y[i] = −t[i] / y[i]`
    fn gradient<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> V {
        map_pairs(output, target, |y, t| -t / clamp_positive(y))
    }
}
