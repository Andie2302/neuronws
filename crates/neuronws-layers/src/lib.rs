//! Neuronale Netze ohne Betriebssystem: `no_std`, ohne `alloc`, ohne `std`, ohne fremde Bibliotheken.
//!
//! Alle Größen stehen im Typ, ein Netz liegt komplett auf dem Stack oder in einer `static`. Die
//! README im Hauptordner des Projekts erklärt das Ganze ausführlicher, hier steht der Überblick.
//!
//! # Die vier Ideen
//!
//! 1. [`block::Operation`] ist die reine Mathematik einer Schicht: `forward` und `backward`.
//! 2. [`block::Block`] macht aus jeder Operation eine Schicht mit Parametern, Gradienten und
//!    Zwischenspeichern und allen Fähigkeiten aus [`traits`] (`ForwardPath`, `BackwardPathMut`,
//!    `Trainable`, ...).
//! 3. [`chain::Chain`] und [`chain::LayerExt::then`] verketten Schichten. Die Typen sorgen dafür, dass
//!    Ausgabe und nächste Eingabe zusammenpassen.
//! 4. [`trainer::Trainer`] bündelt, wie gelernt wird: Fehlerfunktion ([`loss`]), Optimierer
//!    ([`optimizer`]) und Batchgröße.
//!
//! Eingaben, Ausgaben und Parameter sind [`values::Values`]: beliebig verschachtelte Arrays und Tupel
//! fester Größe.
//!
//! # Features
//!
//! Ohne Feature ist die Bibliothek reines `core`. Alles Weitere steht hinter einem Schalter:
//!
//! | Feature | Bringt |
//! |---|---|
//! | `libm` | `exp`, `tanh`, `ln`, `sqrt`, `cos`, `sin` aus der Bibliothek `libm` (rein Rust, `no_std`). |
//! | `std` | Dieselben Funktionen aus der Standardbibliothek. Sind `std` und `libm` an, gewinnt `libm`. |
//! | `alloc` | Alles mit Heap: [`Box<M>`](https://doc.rust-lang.org/alloc/boxed/struct.Box.html) als Schicht, siehe `boxed`. |
//!
//! Was Mathe-Funktionen braucht (Sigmoid, Tanh, Adam, Softmax, Xavier, ...), steht in der
//! Dokumentation des jeweiligen Typs und fehlt ohne `libm` oder `std`.
//!
//! # Ein Netz ohne Mathe-Bibliothek
//!
//! ```
//! use neuronws_layers::activation::Relu;
//! use neuronws_layers::activation_layer::ActivationLayer;
//! use neuronws_layers::chain::LayerExt;
//! use neuronws_layers::dense_layer::DenseLayer;
//! use neuronws_layers::loss::MeanSquaredError;
//! use neuronws_layers::optimizer::Sgd;
//! use neuronws_layers::trainer::Trainer;
//!
//! // 1 Eingang → 3 Neuronen (ReLU) → 1 Ausgang. Das geht auch ohne jedes Feature.
//! let mut net = DenseLayer::new([[1.0], [-1.0], [0.5]], [0.0, 1.0, 0.0])
//!     .then(ActivationLayer::<3, _>::new(Relu))
//!     .then(DenseLayer::new([[0.1, 0.1, 0.1]], [0.0]));
//! let data = [([1.0], [2.0]), ([2.0], [4.0]), ([3.0], [6.0])];
//!
//! let mut trainer = Trainer::new(MeanSquaredError, Sgd::for_model(&net, 0.01), 3);
//! let before: f32 = trainer.evaluate(&net, &data);
//! let after: f32 = trainer.fit(&mut net, &data, 200);
//! assert!(after < before);
//! ```
//!
//! # Wo man was findet
//!
//! - **Schichten:** [`dense_layer`], [`diagonal`], [`conv`], [`mod@conv1d`], [`conv_transpose`], [`pooling`],
//!   `norm`, [`dropout`], [`residual`], [`recurrent`], [`sequence`], [`reshape`], [`standardize`],
//!   `softmax_layer`, [`each`], [`neurons`].
//! - **Bausteine zum Austauschen:** [`activation`], [`activation_layer`], [`loss`], [`init`].
//! - **Lernen:** [`optimizer`], `adam`, `adaptive`, [`wrappers`] (Clipping, L2), [`schedule`],
//!   [`trainer`], [`training`], [`data`], [`early_stopping`], [`metrics`].
//! - **Daten und Zufall:** [`datasets`], [`random`].
//! - **Speichern:** [`persist`] (ohne `std` und ohne `alloc`).
//! - **Prüfen:** [`gradcheck`] für den numerischen Gradienten-Check eigener Schichten.
//! - **Grundlagen:** [`scalar`] (`f32` und `f64`), [`values`], [`traits`], [`block`].

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
