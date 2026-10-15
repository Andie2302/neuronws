#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod activation;
pub mod activation_layer;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod adam;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod adaptive;
pub mod block;
#[cfg(feature = "alloc")]
pub mod boxed;
pub mod chain;
pub mod conv;
pub mod dense_layer;
pub mod diagonal;
pub mod dropout;
pub mod each;
pub mod gradcheck;
pub mod init;
pub mod loss;
pub mod neurons;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod norm;
pub mod optimizer;
pub mod pooling;
pub mod random;
pub mod reshape;
pub mod residual;
pub mod scalar;
pub mod schedule;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod softmax_layer;
pub mod standardize;
pub mod trainer;
pub mod training;
pub mod traits;
pub mod values;
pub mod wrappers;
