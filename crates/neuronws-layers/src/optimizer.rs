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
/// `P` ist die Anzahl der Parameter des Modells. Die Parameter werden in einer festen
/// Reihenfolge besucht, und `velocity[i]` gehört zum `i`-ten.
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
    /// Wenn `P` nicht der Anzahl der Parameter des Modells entspricht.
    fn step<M: Trainable<Scalar = T>>(&mut self, model: &mut M, batch_size: usize) {
        let scale = self.learning_rate / T::from_usize(batch_size);
        let momentum = self.momentum;
        let velocity = &mut self.velocity;
        let mut count = 0;

        model.visit_parameters(&mut |parameter, gradient| {
            assert!(
                count < P,
                "Momentum<{P}>: das Modell hat mehr als {P} Parameter"
            );
            velocity[count] = momentum * velocity[count] - scale * gradient;
            *parameter += velocity[count];
            count += 1;
        });

        assert_eq!(count, P, "Momentum<{P}>: das Modell hat {count} Parameter");
        model.zero_grad();
    }
}
