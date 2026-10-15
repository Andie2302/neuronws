use core::fmt;

use crate::scalar::Scalar;
use crate::traits::Trainable;
use crate::values::Values;

/// Ein Optimierer: macht aus den gesammelten Gradienten einen Lernschritt.
///
/// Die Gradienten sind über `batch_size` Beispiele aufsummiert. Der Optimierer nimmt den
/// Mittelwert, ändert die Parameter und setzt die Gradienten danach zurück.
/// Die Lernrate gehört dem Optimierer, die Schichten wissen nichts davon.
///
/// Der Optimierer ist für ein bestimmtes Modell `M` gemacht. Hat er ein Gedächtnis, ist es genau
/// so groß wie die Parameter von `M` (`M::Shape`), eine falsche Größe gibt es nicht.
pub trait Optimizer<M: Trainable> {
    fn step(&mut self, model: &mut M, batch_size: usize);
}

/// Stochastic Gradient Descent: `p ← p − η · g`.
#[derive(Debug, Clone)]
pub struct Sgd<T: Scalar = f32> {
    learning_rate: T,
}

impl<T: Scalar> Sgd<T> {
    pub const fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }
}

impl<M: Trainable> Optimizer<M> for Sgd<M::Scalar> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        let scale = self.learning_rate / M::Scalar::from_usize(batch_size);
        model.visit_parameters(&mut |parameter, gradient| *parameter -= scale * gradient);
        model.zero_grad();
    }
}

/// SGD mit Impuls: `v ← μ v − η g`, `p ← p + v`.
///
/// Jeder Parameter merkt sich seine bisherige "Geschwindigkeit" `v`. Das glättet Zickzack-Wege
/// und beschleunigt das Lernen in Richtungen, in denen der Gradient immer gleich zeigt.
///
/// Das Gedächtnis hat die Form der Parameter des Modells (`M::Shape`). Die Parameter werden in
/// fester Reihenfolge besucht, und `velocity[i]` gehört zum `i`-ten.
///
/// ```
/// use neuronws_layers::activation::Relu;
/// use neuronws_layers::activation_layer::ActivationLayer;
/// use neuronws_layers::chain::Chain;
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::optimizer::Momentum;
///
/// type Net = Chain<DenseLayer<2, 3>, Chain<ActivationLayer<3, Relu>, DenseLayer<3, 1>>>;
///
/// // Entweder über den Typ des Netzes ...
/// let _momentum = Momentum::<Net>::new(0.01, 0.9);
///
/// // ... oder über ein vorhandenes Netz, ohne den Typ hinzuschreiben.
/// let net: Net = Chain::new(
///     DenseLayer::new([[0.0; 2]; 3], [0.0; 3]),
///     Chain::new(ActivationLayer::new(Relu), DenseLayer::new([[0.0; 3]], [0.0])),
/// );
/// let _momentum = Momentum::for_model(&net, 0.01, 0.9);
/// ```
///
/// Ein Optimierer passt nur zum Modell, für das er gebaut wurde. Das prüft der Compiler:
///
/// ```compile_fail
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::optimizer::{Momentum, Optimizer};
///
/// let mut small = DenseLayer::new([[0.0]], [0.0]);
/// let mut momentum = Momentum::<DenseLayer<2, 2>>::new(0.01, 0.9);
/// momentum.step(&mut small, 1); // DenseLayer<1, 1> hat andere Parameter als DenseLayer<2, 2>
/// ```
pub struct Momentum<M: Trainable> {
    learning_rate: M::Scalar,
    momentum: M::Scalar,
    velocity: M::Shape,
}

impl<M: Trainable> Momentum<M> {
    pub fn new(learning_rate: M::Scalar, momentum: M::Scalar) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: <M::Shape as Values<M::Scalar>>::zeros(),
        }
    }

    /// ... oder über ein vorhandenes Netz, ohne den Typ hinzuschreiben.
    pub fn for_model(_model: &M, learning_rate: M::Scalar, momentum: M::Scalar) -> Self {
        Self::new(learning_rate, momentum)
    }
}

impl<M: Trainable> Optimizer<M> for Momentum<M> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        let scale = self.learning_rate / M::Scalar::from_usize(batch_size);
        let momentum = self.momentum;
        let velocity = &mut self.velocity;
        let mut index = 0;

        model.visit_parameters(&mut |parameter, gradient| {
            let v = velocity.value_mut(index);
            *v = momentum * *v - scale * gradient;
            *parameter += *v;
            index += 1;
        });

        model.zero_grad();
    }
}

impl<M: Trainable> Clone for Momentum<M> {
    fn clone(&self) -> Self {
        Self {
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            velocity: self.velocity,
        }
    }
}

impl<M: Trainable> fmt::Debug for Momentum<M>
where
    M::Scalar: fmt::Debug,
    M::Shape: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Momentum")
            .field("learning_rate", &self.learning_rate)
            .field("momentum", &self.momentum)
            .field("velocity", &self.velocity)
            .finish()
    }
}
