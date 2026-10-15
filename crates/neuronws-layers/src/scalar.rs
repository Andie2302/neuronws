use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// Ein Zahlentyp, mit dem die Schichten rechnen: `f32` oder `f64`.
///
/// `exp`, `tanh`, `sqrt`, `ln`, `cos` und `sin` gibt es in `core` nicht. Sie sind nur mit dem
/// Feature `std` (dann aus `std`) oder `libm` (dann aus `libm`) verfügbar. Sind beide Features an,
/// gewinnt `libm`.
///
/// Alles andere (`abs`, `min`, `max`, `clamp`, `sign`) lässt sich mit Vergleichen bauen und steht
/// deshalb immer zur Verfügung, auch ohne `std` und ohne `libm`.
pub trait Scalar:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;

    /// Wie viele Bytes die Zahl in der Binärdarstellung belegt (4 für `f32`, 8 für `f64`).
    /// Siehe [`write_le`](Self::write_le) und [`read_le`](Self::read_le).
    const BYTES: usize;

    fn from_f32(value: f32) -> Self;
    fn from_f64(value: f64) -> Self;
    fn from_usize(value: usize) -> Self;

    /// Für Ausgaben und Diagnose: der Wert als `f64` (bei `f32` verlustfrei).
    fn to_f64(self) -> f64;

    /// Schreibt die Zahl als [`BYTES`](Self::BYTES) Bytes, niedrigstes Byte zuerst (little endian).
    ///
    /// # Panics
    /// Wenn `out` kürzer als `BYTES` ist.
    fn write_le(self, out: &mut [u8]);

    /// Gegenstück zu [`write_le`](Self::write_le): liest die ersten `BYTES` Bytes.
    ///
    /// # Panics
    /// Wenn `bytes` kürzer als `BYTES` ist.
    fn read_le(bytes: &[u8]) -> Self;

    /// Der Betrag `|x|`.
    fn abs(self) -> Self {
        if self < Self::ZERO { -self } else { self }
    }

    /// Das Vorzeichen: 1, −1 oder 0 (auch für 0 und `NaN`).
    fn sign(self) -> Self {
        if self > Self::ZERO {
            Self::ONE
        } else if self < Self::ZERO {
            -Self::ONE
        } else {
            Self::ZERO
        }
    }

    fn min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }

    fn max(self, other: Self) -> Self {
        if other > self { other } else { self }
    }

    /// Begrenzt `self` auf den Bereich `[low, high]`.
    fn clamp(self, low: Self, high: Self) -> Self {
        self.max(low).min(high)
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    fn exp(self) -> Self;
    #[cfg(any(feature = "std", feature = "libm"))]
    fn tanh(self) -> Self;
    #[cfg(any(feature = "std", feature = "libm"))]
    fn sqrt(self) -> Self;
    #[cfg(any(feature = "std", feature = "libm"))]
    fn ln(self) -> Self;
    #[cfg(any(feature = "std", feature = "libm"))]
    fn cos(self) -> Self;
    #[cfg(any(feature = "std", feature = "libm"))]
    fn sin(self) -> Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const BYTES: usize = 4;

    fn from_f32(value: f32) -> Self {
        value
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn from_usize(value: usize) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn write_le(self, out: &mut [u8]) {
        out[..4].copy_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &[u8]) -> Self {
        let mut buffer = [0; 4];
        buffer.copy_from_slice(&bytes[..4]);
        f32::from_le_bytes(buffer)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[cfg(feature = "libm")]
    fn exp(self) -> Self {
        libm::expf(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn tanh(self) -> Self {
        f32::tanh(self)
    }
    #[cfg(feature = "libm")]
    fn tanh(self) -> Self {
        libm::tanhf(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[cfg(feature = "libm")]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn ln(self) -> Self {
        f32::ln(self)
    }
    #[cfg(feature = "libm")]
    fn ln(self) -> Self {
        libm::logf(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn cos(self) -> Self {
        f32::cos(self)
    }
    #[cfg(feature = "libm")]
    fn cos(self) -> Self {
        libm::cosf(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn sin(self) -> Self {
        f32::sin(self)
    }
    #[cfg(feature = "libm")]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const BYTES: usize = 8;

    fn from_f32(value: f32) -> Self {
        f64::from(value)
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn from_usize(value: usize) -> Self {
        value as f64
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn write_le(self, out: &mut [u8]) {
        out[..8].copy_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &[u8]) -> Self {
        let mut buffer = [0; 8];
        buffer.copy_from_slice(&bytes[..8]);
        f64::from_le_bytes(buffer)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[cfg(feature = "libm")]
    fn exp(self) -> Self {
        libm::exp(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[cfg(feature = "libm")]
    fn tanh(self) -> Self {
        libm::tanh(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[cfg(feature = "libm")]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[cfg(feature = "libm")]
    fn ln(self) -> Self {
        libm::log(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[cfg(feature = "libm")]
    fn cos(self) -> Self {
        libm::cos(self)
    }

    #[cfg(all(feature = "std", not(feature = "libm")))]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[cfg(feature = "libm")]
    fn sin(self) -> Self {
        libm::sin(self)
    }
}
