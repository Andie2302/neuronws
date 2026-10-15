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
