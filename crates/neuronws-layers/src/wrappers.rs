//! Hüllen um einen Optimierer: Sie verändern die Gradienten, bevor der eigentliche Optimierer
//! sie sieht, und reichen den Lernschritt dann weiter.
//!
//! ```
//! use neuronws_layers::dense_layer::DenseLayer;
//! use neuronws_layers::optimizer::Sgd;
//! use neuronws_layers::wrappers::{ClipValue, L2};
//!
//! let layer = DenseLayer::new([[0.0; 2]], [0.0]);
//! // Erst L2-Regularisierung, dann Werte begrenzen, dann SGD. Jede Hülle ist selbst ein
//! // Optimierer und lässt sich mit der nächsten umhüllen.
//! let _optimizer = L2::new(ClipValue::new(Sgd::new(0.1), 1.0), 0.01);
//! # let _ = layer;
//! ```
//!
//! Die Gradienten im Modell sind über den ganzen Batch **aufsummiert**. Die Hüllen rechnen
//! deshalb so, als ginge es um den **Mittelwert** (Summe geteilt durch `batch_size`), denn der
//! ist es, den man begrenzen oder regularisieren will. So bleiben die Einstellungen unabhängig
//! von der Batchgröße.

use crate::optimizer::{HasLearningRate, Optimizer, batch_divisor};
use crate::scalar::Scalar;
use crate::traits::Trainable;

/// Begrenzt jeden Wert des (gemittelten) Gradienten auf `[−limit, limit]`.
///
/// Verhindert, dass ein einzelner Ausreißer im Gradienten einen riesigen Sprung auslöst. Die
/// Richtung kann sich dabei ändern, weil jeder Wert für sich begrenzt wird. Will man die Richtung
/// behalten, ist [`ClipNorm`] besser. Braucht keine Mathe-Funktionen.
#[derive(Debug, Clone)]
pub struct ClipValue<O> {
    inner: O,
    limit: f32,
}

impl<O> ClipValue<O> {
    pub const fn new(inner: O, limit: f32) -> Self {
        Self { inner, limit }
    }

    pub const fn inner(&self) -> &O {
        &self.inner
    }

    pub const fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    pub fn into_inner(self) -> O {
        self.inner
    }
}

impl<M: Trainable, O: Optimizer<M>> Optimizer<M> for ClipValue<O> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        // Mittelwert in [−limit, limit] heißt Summe in [−limit·B, limit·B].
        let bound = M::Scalar::from_f32(self.limit) * batch_divisor::<M::Scalar>(batch_size);
        model.visit_parameters(&mut |_, gradient| *gradient = gradient.clamp(-bound, bound));
        self.inner.step(model, batch_size);
    }
}

impl<T: Scalar, O: HasLearningRate<T>> HasLearningRate<T> for ClipValue<O> {
    fn learning_rate(&self) -> T {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, learning_rate: T) {
        self.inner.set_learning_rate(learning_rate);
    }
}

/// Begrenzt die **Länge** des ganzen (gemittelten) Gradienten auf höchstens `max_norm`.
///
/// Ist die Länge `‖g‖ = √(Σ g²)` über alle Parameter größer, wird der Gradient auf diese Länge
/// gestaucht. Seine Richtung bleibt dabei erhalten. Das ist die übliche Gegenmaßnahme gegen
/// explodierende Gradienten, vor allem bei rückgekoppelten Netzen (RNN). Braucht die Wurzel,
/// deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone)]
pub struct ClipNorm<O> {
    inner: O,
    max_norm: f32,
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<O> ClipNorm<O> {
    pub const fn new(inner: O, max_norm: f32) -> Self {
        Self { inner, max_norm }
    }

    pub const fn inner(&self) -> &O {
        &self.inner
    }

    pub const fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    pub fn into_inner(self) -> O {
        self.inner
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<M: Trainable, O: Optimizer<M>> Optimizer<M> for ClipNorm<O> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        let batch = batch_divisor::<M::Scalar>(batch_size);
        let mut sum_of_squares = M::Scalar::ZERO;
        model.inspect_parameters(&mut |_, gradient| sum_of_squares += gradient * gradient);
        let norm = sum_of_squares.sqrt() / batch;

        let max_norm = M::Scalar::from_f32(self.max_norm);
        if norm > max_norm {
            let factor = max_norm / norm;
            model.visit_parameters(&mut |_, gradient| *gradient = *gradient * factor);
        }
        self.inner.step(model, batch_size);
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T: Scalar, O: HasLearningRate<T>> HasLearningRate<T> for ClipNorm<O> {
    fn learning_rate(&self) -> T {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, learning_rate: T) {
        self.inner.set_learning_rate(learning_rate);
    }
}

/// L2-Regularisierung: bestraft große Gewichte, indem `λ · p` zum Gradienten jedes Parameters
/// dazukommt.
///
/// Das entspricht dem Zusatzterm `λ/2 · Σ p²` in der Fehlerfunktion. Die Gewichte werden bei
/// jedem Schritt ein Stück Richtung 0 gezogen, das Netz bevorzugt kleine Gewichte und passt sich
/// weniger an Rauschen an.
///
/// Die Hülle wirkt auf **alle** Parameter, auch auf die Bias-Werte. Bei einfachem SGD ist das
/// dasselbe wie "Weight Decay". Bei [`AdamW`](crate::adam::AdamW) gibt es dafür einen eigenen,
/// entkoppelten Weg (`weight_decay`), der meistens besser ist. Hier zählt der Term wie ein
/// normaler Gradient und geht durch die Adam-Skalierung. Braucht keine Mathe-Funktionen.
#[derive(Debug, Clone)]
pub struct L2<O> {
    inner: O,
    lambda: f32,
}

impl<O> L2<O> {
    pub const fn new(inner: O, lambda: f32) -> Self {
        Self { inner, lambda }
    }

    pub const fn inner(&self) -> &O {
        &self.inner
    }

    pub const fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    pub fn into_inner(self) -> O {
        self.inner
    }
}

impl<M: Trainable, O: Optimizer<M>> Optimizer<M> for L2<O> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        // Der Gradient im Modell ist eine Summe über `B` Beispiele. Der Zusatz `λ p` gehört zum
        // Mittelwert, deshalb kommt `λ · B · p` zur Summe.
        let strength = M::Scalar::from_f32(self.lambda) * batch_divisor::<M::Scalar>(batch_size);
        model.visit_parameters(&mut |parameter, gradient| *gradient += strength * *parameter);
        self.inner.step(model, batch_size);
    }
}

impl<T: Scalar, O: HasLearningRate<T>> HasLearningRate<T> for L2<O> {
    fn learning_rate(&self) -> T {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, learning_rate: T) {
        self.inner.set_learning_rate(learning_rate);
    }
}
