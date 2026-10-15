use crate::scalar::Scalar;
use crate::traits::Trainable;

/// Stochastic Gradient Descent: `p ← p − η · g`.
///
/// Die Lernrate gehört dem Optimierer, die Schichten wissen nichts davon.
pub struct Sgd<T: Scalar = f32> {
    learning_rate: T,
}

impl<T: Scalar> Sgd<T> {
    pub const fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }

    /// Ein Lernschritt. Die Gradienten sind über `batch_size` Beispiele aufsummiert,
    /// deshalb wird der Mittelwert genommen. Danach werden sie zurückgesetzt.
    pub fn step<M: Trainable<Scalar = T>>(&self, model: &mut M, batch_size: usize) {
        let scale = self.learning_rate / T::from_usize(batch_size);
        model.visit_parameters(&mut |parameter, gradient| *parameter -= scale * gradient);
        model.zero_grad();
    }
}
