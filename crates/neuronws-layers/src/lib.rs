#![cfg_attr(not(feature = "std"), no_std)]
pub mod activation;
pub mod activation_layer;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod adam;
pub mod block;
pub mod chain;
pub mod dense_layer;
pub mod gradcheck;
pub mod init;
pub mod loss;
pub mod neurons;
pub mod optimizer;
pub mod random;
pub mod scalar;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod softmax_layer;
pub mod standardize;
pub mod training;
pub mod traits;
pub mod values;
