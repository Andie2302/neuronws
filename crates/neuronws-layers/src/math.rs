//! Mathe-Funktionen, die es in `core` nicht gibt.
//!
//! Mit dem Feature `std` kommen sie aus `std`, sonst aus `libm`.

#[cfg(feature = "std")]
pub fn exp(x: f32) -> f32 {
    x.exp()
}

#[cfg(feature = "std")]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[cfg(all(feature = "libm", not(feature = "std")))]
pub fn exp(x: f32) -> f32 {
    libm::expf(x)
}

#[cfg(all(feature = "libm", not(feature = "std")))]
pub fn tanh(x: f32) -> f32 {
    libm::tanhf(x)
}

#[cfg(feature = "std")]
pub fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

#[cfg(all(feature = "libm", not(feature = "std")))]
pub fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}
