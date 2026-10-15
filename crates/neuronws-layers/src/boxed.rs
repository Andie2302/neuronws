//! `Box<M>` als Schicht: Große Netze auf dem Heap statt auf dem Stack. Braucht das Feature `alloc`.
//!
//! Jeder [`Block`](crate::block::Block) hält Eingabe, Ausgabe, Parameter und Gradienten als
//! Arrays. Bei Bildern kommt schnell einiges zusammen, und der Stack ist klein (2 MB in einem
//! Test-Thread, oft nur wenige KB auf einem Mikrocontroller). Ein `Box<M>` legt die Schicht auf
//! den Heap und verhält sich sonst genau wie `M`: Es ist selbst ein `Layer` und lässt sich
//! verketten und trainieren. Auch Optimierer lassen sich boxen.
//!
//! ```
//! # #[cfg(feature = "alloc")]
//! # {
//! use neuronws_layers::activation::Relu;
//! use neuronws_layers::activation_layer::ActivationLayer;
//! use neuronws_layers::chain::LayerExt;
//! use neuronws_layers::dense_layer::DenseLayer;
//! use neuronws_layers::traits::ForwardPath;
//!
//! // Jede Schicht einzeln auf den Heap, die Kette selbst bleibt winzig (nur Zeiger).
//! let net = Box::new(DenseLayer::new([[1.0, -1.0]], [0.0]))
//!     .then(Box::new(ActivationLayer::<1, _>::new(Relu)));
//! assert_eq!(net.forward(&[3.0, 1.0]), [2.0]);
//! # }
//! ```
//!
//! Gebaut wird eine Schicht trotzdem zuerst auf dem Stack und dann verschoben. Der Stack muss
//! also die **größte einzelne** Schicht aushalten, nicht das ganze Netz. Ohne `unsafe` (das ist
//! im ganzen Projekt verboten) geht es nicht anders.
//!
//! # Wie viel Stack braucht ein Netz?
//!
//! Gemessen mit `examples/stack_budget.rs`: Das Programm startet sich für jedes Netz mit
//! verschieden großen Stacks und sucht den kleinsten, bei dem ein Lernschritt (vorwärts,
//! rückwärts, Optimierer-Schritt mit AdamW) noch läuft. "Größe" ist `size_of` von Netz und
//! Optimierer zusammen. Alle Zahlen für `f32`, auf dem Rechner, auf dem das geschrieben wurde
//! (x86-64):
//!
//! ```text
//!                                    Größe     Stack mindestens
//!                                    im RAM    Release    Debug
//! Dense 784 → 64 → 10                 812 KB    1052 KB   4255 KB
//! Faltung  8 und 16 Kanäle            276 KB     591 KB   2008 KB
//! Faltung 16 und 32 Kanäle            583 KB    1193 KB   4116 KB
//! Faltung 32 und 64 Kanäle           1304 KB    2575 KB   8817 KB
//! Faltung 64 und 128 Kanäle          3177 KB    5852 KB  20211 KB
//! 32/64, jede Schicht in einer Box      0 KB     517 KB   2063 KB
//! 64/128, jede Schicht in einer Box     0 KB    1459 KB   6122 KB
//! ```
//!
//! Zum Vergleich: Ein Test-Thread hat 2048 KB Stack, der Haupt-Thread meist 8192 KB.
//!
//! Daraus folgt:
//!
//! - **Faustregel:** Im Release-Build braucht ein Lernschritt etwa doppelt so viel Stack, wie
//!   Netz und Optimierer groß sind. Im Debug-Build das Vier- bis Sechsfache, weil dort jede
//!   Rückgabe und jedes `.then(...)` das Netz kopiert und der Compiler die Kopien nicht
//!   zusammenlegt. Das Netz `32/64` ist 1,3 MB groß und läuft in einem Test-Thread (2 MB) schon
//!   nicht mehr.
//! - Jede Schicht legt Eingabe, Ausgabe und Gradient der Eingabe als eigene Arrays an, bei Bildern
//!   also dreimal die Größe der Aktivierung. Der Optimierer kommt mit ein bis zwei Kopien der
//!   Parameter dazu (Momentum eine, Adam zwei).
//! - Mit einer `Box` pro Schicht bleibt nur noch die größte einzelne Schicht (und der
//!   Zwischenwert beim Rechnen) auf dem Stack. Das bringt im Release-Build etwa den Faktor 4 bis 5.
//! - Ein größerer Stack hilft mit `std` sofort: `std::thread::Builder::new().stack_size(...)`.
//!
//! **Ohne `alloc` und ohne `std`** (Mikrocontroller) gibt es nur den Weg, das Netz klein zu halten
//! und seine Größe vorher zu kennen: `core::mem::size_of::<Netz>()` und
//! `core::mem::size_of_val(&optimizer)` sind exakt, und ein `static` im Flash oder RAM hält Netz
//! und Optimierer außerhalb des Stacks (ein `static mut` bräuchte `unsafe`, das dieses Projekt
//! verbietet, in einer eigenen Anwendung hilft ein Zellentyp aus deren Laufzeit).

use alloc::boxed::Box;

use crate::optimizer::{HasLearningRate, Optimizer};
use crate::scalar::Scalar;
use crate::traits::{
    BackwardData, BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer, Trainable,
};

impl<L: Layer> Layer for Box<L> {}

impl<L: ForwardData> ForwardData for Box<L> {
    type ForwardInput = L::ForwardInput;
    type ForwardOutput = L::ForwardOutput;
}

impl<L: ForwardPath> ForwardPath for Box<L> {
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        (**self).forward(input)
    }
}

impl<L: ForwardPathMut> ForwardPathMut for Box<L> {
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        (**self).forward_mut(input)
    }
}

impl<L: BackwardData> BackwardData for Box<L> {
    type BackwardInput = L::BackwardInput;
    type BackwardOutput = L::BackwardOutput;
}

impl<L: BackwardPathMut> BackwardPathMut for Box<L> {
    fn backward_mut(&mut self, input: &Self::BackwardInput) -> &Self::BackwardOutput {
        (**self).backward_mut(input)
    }
}

impl<L: Trainable> Trainable for Box<L> {
    type Scalar = L::Scalar;
    type Shape = L::Shape;

    fn visit_parameters(&mut self, visitor: &mut impl FnMut(&mut Self::Scalar, &mut Self::Scalar)) {
        (**self).visit_parameters(visitor);
    }

    fn inspect_parameters(&self, visitor: &mut impl FnMut(Self::Scalar, Self::Scalar)) {
        (**self).inspect_parameters(visitor);
    }

    fn zero_grad(&mut self) {
        (**self).zero_grad();
    }
}

/// Ein geboxter Optimierer ist ein Optimierer. Sein Gedächtnis (`m`, `v`, ...) liegt dann auf dem
/// Heap.
impl<M: Trainable, O: Optimizer<M>> Optimizer<M> for Box<O> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        (**self).step(model, batch_size);
    }
}

impl<T: Scalar, O: HasLearningRate<T>> HasLearningRate<T> for Box<O> {
    fn learning_rate(&self) -> T {
        (**self).learning_rate()
    }

    fn set_learning_rate(&mut self, learning_rate: T) {
        (**self).set_learning_rate(learning_rate);
    }
}
