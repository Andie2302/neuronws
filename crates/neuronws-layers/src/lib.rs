#![cfg_attr(not(feature = "std"), no_std)]
pub mod activation;
pub mod activation_layer;
pub mod chain;
pub mod dense_layer;
pub mod loss;
#[cfg(any(feature = "std", feature = "libm"))]
mod math;
pub mod optimizer;
pub mod random;
pub mod traits;
