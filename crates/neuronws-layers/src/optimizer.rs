use crate::scalar::Scalar;
use crate::traits::Trainable;

/// Ein Optimierer: macht aus den gesammelten Gradienten einen Lernschritt.
///
/// Die Gradienten sind über `batch_size` Beispiele aufsummiert. Der Optimierer nimmt den
/// Mittelwert, ändert die Parameter und setzt die Gradienten danach zurück.
/// Die Lernrate gehört dem Optimierer, die Schichten wissen nichts davon.
pub trait Optimizer<T: Scalar> {
    fn step<M: Trainable<Scalar = T>>(&mut self, model: &mut M, batch_size: usize);
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

impl<T: Scalar> Optimizer<T> for Sgd<T> {
    fn step<M: Trainable<Scalar = T>>(&mut self, model: &mut M, batch_size: usize) {
        let scale = self.learning_rate / T::from_usize(batch_size);
        model.visit_parameters(&mut |parameter, gradient| *parameter -= scale * gradient);
        model.zero_grad();
    }
}

/// SGD mit Impuls: `v ← μ v − η g`, `p ← p + v`.
///
/// Jeder Parameter merkt sich seine bisherige "Geschwindigkeit" `v`. Das glättet Zickzack-Wege
/// und beschleunigt das Lernen in Richtungen, in denen der Gradient immer gleich zeigt.
///
/// `P` ist die Anzahl der Parameter des Modells, also `<Modell as Trainable>::PARAMETER_COUNT`.
/// Die Parameter werden in einer festen Reihenfolge besucht, und `velocity[i]` gehört zum `i`-ten.
///
/// ```
/// use neuronws_layers::activation::Relu;
/// use neuronws_layers::activation_layer::ActivationLayer;
/// use neuronws_layers::chain::Chain;
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::optimizer::Momentum;
/// use neuronws_layers::traits::Trainable;
///
/// type Net = Chain<DenseLayer<2, 3>, Chain<ActivationLayer<3, Relu>, DenseLayer<3, 1>>>;
/// const P: usize = <Net as Trainable>::PARAMETER_COUNT; // 2*3 + 3 + 0 + 3*1 + 1 = 13
///
/// let _momentum = Momentum::<P>::new(0.01, 0.9);
/// assert_eq!(P, 13);
/// ```
#[derive(Debug, Clone)]
pub struct Momentum<const P: usize, T: Scalar = f32> {
    learning_rate: T,
    momentum: T,
    velocity: [T; P],
}

impl<const P: usize, T: Scalar> Momentum<P, T> {
    pub const fn new(learning_rate: T, momentum: T) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: [T::ZERO; P],
        }
    }
}

impl<const P: usize, T: Scalar> Optimizer<T> for Momentum<P, T> {
    /// # Panics
    /// Wenn `P` nicht der Anzahl der Parameter des Modells entspricht. Das wird vor dem ersten
    /// Schritt geprüft, es wird also nie ein Teil der Parameter geändert.
    fn step<M: Trainable<Scalar = T>>(&mut self, model: &mut M, batch_size: usize) {
        assert_eq!(
            M::PARAMETER_COUNT,
            P,
            "Momentum<{P}> passt nicht zu einem Modell mit {} Parametern",
            M::PARAMETER_COUNT
        );
        let scale = self.learning_rate / T::from_usize(batch_size);
        let momentum = self.momentum;
        let velocity = &mut self.velocity;
        let mut index = 0;

        model.visit_parameters(&mut |parameter, gradient| {
            velocity[index] = momentum * velocity[index] - scale * gradient;
            *parameter += velocity[index];
            index += 1;
        });

        model.zero_grad();
    }
}
