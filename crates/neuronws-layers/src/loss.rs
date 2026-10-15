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

/// Mittlerer absoluter Fehler: `L = (1/N) Σ |y − t|`.
///
/// Anders als beim quadratischen Fehler zählt ein Ausreißer nur linear, nicht im Quadrat. Der
/// Gradient hat überall den Betrag `1/N`, egal wie weit die Ausgabe danebenliegt. Bei `y = t` ist
/// der Betrag nicht ableitbar, dort wird der Gradient 0 gesetzt.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanAbsoluteError;

impl Loss for MeanAbsoluteError {
    fn value<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> T {
        sum_pairs(output, target, |y, t| (y - t).abs()) / T::from_usize(V::LEN)
    }

    /// `∂L/∂y[i] = sign(y[i] − t[i]) / N`
    fn gradient<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> V {
        let n = T::from_usize(V::LEN);
        map_pairs(output, target, |y, t| (y - t).sign() / n)
    }
}

/// Huber-Verlust: quadratisch nahe am Ziel, linear weit weg.
///
/// ```text
/// pro Wert, mit d = y − t:    ½ d²            für |d| ≤ δ
///                              δ (|d| − ½ δ)   sonst
/// ```
///
/// `L` ist der Mittelwert über alle Werte, `δ` (Standard 1) die Schwelle zwischen beiden Teilen.
/// Das vereint das Beste aus zwei Welten: Kleine Fehler werden wie beim quadratischen Fehler fein
/// ausgeglichen, große Ausreißer ziehen nur mit begrenzter Kraft `δ`.
#[derive(Debug, Clone, Copy)]
pub struct Huber(pub f32);

impl Default for Huber {
    fn default() -> Self {
        Self(1.0)
    }
}

impl Loss for Huber {
    fn value<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> T {
        let delta = T::from_f32(self.0);
        let half = T::from_f32(0.5);
        let sum = sum_pairs(output, target, |y, t| {
            let d = (y - t).abs();
            if d <= delta {
                half * d * d
            } else {
                delta * (d - half * delta)
            }
        });
        sum / T::from_usize(V::LEN)
    }

    /// `∂L/∂y[i] = d / N` für `|d| ≤ δ`, sonst `δ · sign(d) / N`
    fn gradient<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> V {
        let delta = T::from_f32(self.0);
        let n = T::from_usize(V::LEN);
        map_pairs(output, target, |y, t| {
            let d = y - t;
            d.clamp(-delta, delta) / n
        })
    }
}

/// Softmax und Kreuzentropie in einem Schritt: `L = −Σ t[i] ln(softmax(z)[i])`.
///
/// Die Eingabe sind die rohen Werte `z` (die "Logits") direkt aus der letzten Dense-Schicht, ohne
/// `SoftmaxLayer` dahinter. Das ist **numerisch stabiler** als erst Softmax und dann
/// [`CategoricalCrossEntropy`]: Es wird nie ein winziges `y` abgeschnitten oder durch `y` geteilt.
/// Stattdessen gilt mit `m = max(z)`:
///
/// ```text
/// L = ln(Σ e^(z[j] − m)) + m − z[ziel]           (für ein One-Hot-Ziel)
/// ∂L/∂z[i] = softmax(z)[i] − t[i]
/// ```
///
/// Der Gradient ist die einfache Differenz aus Wahrscheinlichkeit und Ziel. Für allgemeine Ziele
/// `t` (auch "weiche" Verteilungen) gilt `L = Σt · (ln Σ e^(z[j] − m) + m) − Σ t[i] z[i]`.
///
/// Zum Vorhersagen reicht `argmax` der Logits. Will man Wahrscheinlichkeiten sehen, hängt man
/// beim Einsatz eine `SoftmaxLayer` an. Braucht `exp` und `ln`, deshalb nur mit dem Feature `std`
/// oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct SoftmaxCrossEntropy;

/// `m = max(z)` und `ln Σ e^(z[j] − m)` von allen Werten.
#[cfg(any(feature = "std", feature = "libm"))]
fn max_and_log_sum_exp<T: Scalar, V: Values<T>>(logits: &V) -> (T, T) {
    let max = (1..V::LEN).fold(logits.value(0), |max, i| max.max(logits.value(i)));
    let sum = (0..V::LEN).fold(T::ZERO, |sum, i| sum + (logits.value(i) - max).exp());
    (max, sum.ln())
}

#[cfg(any(feature = "std", feature = "libm"))]
impl Loss for SoftmaxCrossEntropy {
    fn value<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> T {
        let (max, log_sum_exp) = max_and_log_sum_exp(output);
        let total_target = (0..V::LEN).fold(T::ZERO, |sum, i| sum + target.value(i));
        let weighted_logits = sum_pairs(output, target, |z, t| t * z);
        total_target * (log_sum_exp + max) - weighted_logits
    }

    /// `∂L/∂z[i] = softmax(z)[i] · Σt − t[i]`
    fn gradient<T: Scalar, V: Values<T>>(&self, output: &V, target: &V) -> V {
        let (max, log_sum_exp) = max_and_log_sum_exp(output);
        let total_target = (0..V::LEN).fold(T::ZERO, |sum, i| sum + target.value(i));
        map_pairs(output, target, |z, t| {
            let probability = (z - max - log_sum_exp).exp();
            probability * total_target - t
        })
    }
}
