use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// Ein Zahlentyp, mit dem die Schichten rechnen: `f32` oder `f64`.
///
/// `exp`, `tanh` und `sqrt` gibt es in `core` nicht. Sie sind nur mit dem Feature `std`
/// (dann aus `std`) oder `libm` (dann aus `libm`) verfügbar.
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

    fn from_f32(value: f32) -> Self;
    fn from_usize(value: usize) -> Self;

    #[cfg(any(feature = "std", feature = "libm"))]
    fn exp(self) -> Self;
    #[cfg(any(feature = "std", feature = "libm"))]
    fn tanh(self) -> Self;
    #[cfg(any(feature = "std", feature = "libm"))]
    fn sqrt(self) -> Self;
    #[cfg(any(feature = "std", feature = "libm"))]
    fn ln(self) -> Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;

    fn from_f32(value: f32) -> Self {
        value
    }

    fn from_usize(value: usize) -> Self {
        value as f32
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[cfg(all(feature = "libm", not(feature = "std")))]
    fn exp(self) -> Self {
        libm::expf(self)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        f32::tanh(self)
    }
    #[cfg(all(feature = "libm", not(feature = "std")))]
    fn tanh(self) -> Self {
        libm::tanhf(self)
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[cfg(all(feature = "libm", not(feature = "std")))]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self {
        f32::ln(self)
    }
    #[cfg(all(feature = "libm", not(feature = "std")))]
    fn ln(self) -> Self {
        libm::logf(self)
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;

    fn from_f32(value: f32) -> Self {
        f64::from(value)
    }

    fn from_usize(value: usize) -> Self {
        value as f64
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[cfg(all(feature = "libm", not(feature = "std")))]
    fn exp(self) -> Self {
        libm::exp(self)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[cfg(all(feature = "libm", not(feature = "std")))]
    fn tanh(self) -> Self {
        libm::tanh(self)
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[cfg(all(feature = "libm", not(feature = "std")))]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[cfg(all(feature = "libm", not(feature = "std")))]
    fn ln(self) -> Self {
        libm::log(self)
    }
}
