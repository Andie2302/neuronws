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
pub mod conv1d;
pub mod conv_transpose;
pub mod data;
pub mod datasets;
pub mod dense_layer;
pub mod diagonal;
pub mod dropout;
pub mod each;
pub mod early_stopping;
pub mod gradcheck;
pub mod init;
pub mod loss;
pub mod metrics;
pub mod neurons;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod norm;
pub mod optimizer;
pub mod persist;
pub mod pooling;
pub mod random;
pub mod recurrent;
pub mod reshape;
pub mod residual;
pub mod scalar;
pub mod schedule;
pub mod sequence;
#[cfg(any(feature = "std", feature = "libm"))]
pub mod softmax_layer;
pub mod standardize;
pub mod trainer;
pub mod training;
pub mod traits;
pub mod values;
pub mod wrappers;

/// Prüft Größenangaben schon bei `cargo check`, nicht erst beim Bauen.
///
/// Schichten wie `Conv2d`, `MaxPool2d`, `AvgPool2d` und `Reshape` prüfen ihre Größen in `new()`.
/// Diese Prüfung läuft erst, wenn der Compiler den konkreten Typ einsetzt, und das passiert bei
/// `cargo check` und in der Anzeige der IDE nicht. Das Makro ruft `new()` stattdessen in einer
/// `const`, und die wertet der Compiler immer sofort aus:
///
/// ```
/// use neuronws_layers::conv::Conv2dValid;
/// use neuronws_layers::pooling::MaxPool2d;
/// use neuronws_layers::reshape::Reshape;
/// use neuronws_layers::values::Image;
///
/// // Einmal an einer zentralen Stelle neben dem Netz aufschreiben:
/// neuronws_layers::check_sizes!(
///     Conv2dValid<1, 4, 6, 6, 3, 3, 4, 4>,
///     MaxPool2d<4, 4, 4, 2, 2, 2>,
///     Reshape<Image<4, 2, 2>, [f32; 16]>,
/// );
/// ```
///
/// Stimmt eine Größe nicht, meldet schon `cargo check` den Fehler:
///
/// ```compile_fail
/// use neuronws_layers::conv::Conv2dValid;
///
/// // 6×6-Bild mit 3×3-Kern ergibt 4×4, nicht 5×5.
/// neuronws_layers::check_sizes!(Conv2dValid<1, 4, 6, 6, 3, 3, 5, 5>);
/// ```
///
/// Die Typen müssen eine `const fn new()` ohne Argumente haben.
#[macro_export]
macro_rules! check_sizes {
    ($($shape:ty),+ $(,)?) => {
        $( const _: $shape = <$shape>::new(); )+
    };
}
