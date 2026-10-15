use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::Values;

/// Wie die Startwerte der Gewichte gewählt werden.
///
/// `fan_in` ist die Zahl der Eingänge eines Neurons, `fan_out` die Zahl der Neuronen der Schicht.
/// Manche Verfahren skalieren die Werte damit, damit `z` am Anfang weder winzig noch riesig ist.
pub trait Initializer {
    /// Ein einzelner Startwert.
    fn sample<T: Scalar>(&self, rng: &mut Rng, fan_in: usize, fan_out: usize) -> T;

    /// Füllt eine ganze Gewichtsgruppe `target`, die als Matrix mit `fan_in` Spalten gelesen
    /// wird (Zeile = ein Neuron, zeilenweise nummeriert wie bei [`Values`]).
    ///
    /// Der Standard zieht jeden Wert einzeln mit [`sample`](Self::sample). Verfahren, bei denen
    /// die Werte einer Matrix zusammenhängen (zum Beispiel `Orthogonal`), überschreiben das.
    fn fill<T: Scalar, V: Values<T>>(
        &self,
        rng: &mut Rng,
        target: &mut V,
        fan_in: usize,
        fan_out: usize,
    ) {
        for index in 0..V::LEN {
            *target.value_mut(index) = self.sample(rng, fan_in, fan_out);
        }
    }
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

/// Normalverteilt mit Mittelwert 0 und der festen Streuung `std`, unabhängig von der Schichtgröße.
///
/// Braucht `ln`, `sqrt`, `cos` und `sin` (Box-Muller, siehe [`Rng::next_normal`]), deshalb nur
/// mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy)]
pub struct Normal(pub f32);

#[cfg(any(feature = "std", feature = "libm"))]
impl Initializer for Normal {
    fn sample<T: Scalar>(&self, rng: &mut Rng, _fan_in: usize, _fan_out: usize) -> T {
        T::from_f32(self.0) * T::from_f32(rng.next_normal())
    }
}

/// Xavier (Glorot) mit Normalverteilung: Streuung `√(2 / (fan_in + fan_out))`.
///
/// Dieselbe Idee wie [`Xavier`], aber mit der Glockenkurve statt der Gleichverteilung. Große
/// Ausreißer sind damit möglich, aber selten. Braucht `ln`, `sqrt`, `cos` und `sin`, deshalb nur
/// mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct XavierNormal;

#[cfg(any(feature = "std", feature = "libm"))]
impl Initializer for XavierNormal {
    fn sample<T: Scalar>(&self, rng: &mut Rng, fan_in: usize, fan_out: usize) -> T {
        let std_dev = (T::TWO / T::from_usize(fan_in + fan_out)).sqrt();
        std_dev * T::from_f32(rng.next_normal())
    }
}

/// He mit Normalverteilung: Streuung `√(2 / fan_in)`.
///
/// Dieselbe Idee wie [`He`], aber mit der Glockenkurve statt der Gleichverteilung. Braucht `ln`,
/// `sqrt`, `cos` und `sin`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct HeNormal;

#[cfg(any(feature = "std", feature = "libm"))]
impl Initializer for HeNormal {
    fn sample<T: Scalar>(&self, rng: &mut Rng, fan_in: usize, _fan_out: usize) -> T {
        let std_dev = (T::TWO / T::from_usize(fan_in)).sqrt();
        std_dev * T::from_f32(rng.next_normal())
    }
}

/// Orthogonale Startwerte: Die Zeilen (oder, bei mehr Zeilen als Spalten, die Spalten) der
/// Gewichtsmatrix stehen senkrecht aufeinander und haben die Länge `gain` (Standard 1).
///
/// Eine orthogonale Matrix dreht und spiegelt nur, sie streckt oder staucht nichts. Ein Signal
/// behält so beim Durchlauf durch viele Schichten (oder viele Zeitschritte in einem RNN) seine
/// Größe, statt zu explodieren oder zu verschwinden.
///
/// Gebaut wird sie aus normalverteilten Zufallszahlen, die Zeile für Zeile mit dem
/// Gram-Schmidt-Verfahren von den vorigen Zeilen "bereinigt" und auf Länge `gain` gebracht
/// werden. Das braucht die ganze Matrix auf einmal und geht deshalb über [`Initializer::fill`].
/// Als einzelner Wert (`sample`) fällt es auf `gain · N(0, 1)/√fan_in` zurück.
///
/// Braucht `ln`, `sqrt`, `cos` und `sin`, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy)]
pub struct Orthogonal(pub f32);

#[cfg(any(feature = "std", feature = "libm"))]
impl Default for Orthogonal {
    fn default() -> Self {
        Self(1.0)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl Initializer for Orthogonal {
    fn sample<T: Scalar>(&self, rng: &mut Rng, fan_in: usize, _fan_out: usize) -> T {
        T::from_f32(self.0) * T::from_f32(rng.next_normal()) / T::from_usize(fan_in).sqrt()
    }

    /// # Panics
    /// Wenn die Zahl der Werte nicht durch `fan_in` teilbar ist.
    fn fill<T: Scalar, V: Values<T>>(
        &self,
        rng: &mut Rng,
        target: &mut V,
        fan_in: usize,
        _fan_out: usize,
    ) {
        if V::LEN == 0 {
            return;
        }
        assert!(
            fan_in > 0 && V::LEN % fan_in == 0,
            "Orthogonal: {} Werte passen nicht in Zeilen zu je {fan_in}",
            V::LEN
        );
        let (rows, cols) = (V::LEN / fan_in, fan_in);
        // Bei mehr Zeilen als Spalten können nur die Spalten orthogonal sein.
        let by_rows = rows <= cols;
        let (lines, line_len) = if by_rows { (rows, cols) } else { (cols, rows) };
        let position = |line: usize, k: usize| {
            if by_rows {
                line * cols + k
            } else {
                k * cols + line
            }
        };
        let gain = T::from_f32(self.0);
        let tiny = T::from_f32(1e-6);

        for line in 0..lines {
            // Fast immer reicht ein Versuch. Die Wiederholung fängt den Fall ab, dass der
            // Zufallsvektor (fast) in dem schon belegten Raum liegt.
            for _attempt in 0..32 {
                for k in 0..line_len {
                    *target.value_mut(position(line, k)) = T::from_f32(rng.next_normal());
                }
                // Zweimal bereinigen: Das gleicht Rundungsfehler aus ("twice is enough").
                for _pass in 0..2 {
                    for earlier in 0..line {
                        let dot = (0..line_len).fold(T::ZERO, |sum, k| {
                            sum + target.value(position(line, k))
                                * target.value(position(earlier, k))
                        });
                        for k in 0..line_len {
                            let projection = dot * target.value(position(earlier, k));
                            *target.value_mut(position(line, k)) -= projection;
                        }
                    }
                }
                let norm = (0..line_len)
                    .fold(T::ZERO, |sum, k| {
                        let x = target.value(position(line, k));
                        sum + x * x
                    })
                    .sqrt();
                if norm > tiny {
                    for k in 0..line_len {
                        *target.value_mut(position(line, k)) =
                            target.value(position(line, k)) / norm;
                    }
                    break;
                }
            }
        }

        // Erst am Ende auf die gewünschte Länge bringen: Das Bereinigen oben rechnet mit
        // Einheitsvektoren.
        for index in 0..V::LEN {
            *target.value_mut(index) = target.value(index) * gain;
        }
    }
}
