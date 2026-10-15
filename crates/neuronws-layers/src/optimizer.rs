use crate::traits::Trainable;

/// Stochastic Gradient Descent: `p ← p − η · g`.
///
/// Die Lernrate gehört dem Optimierer, die Schichten wissen nichts davon.
pub struct Sgd {
    learning_rate: f32,
}

impl Sgd {
    pub const fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Ein Lernschritt. Die Gradienten sind über `batch_size` Beispiele aufsummiert,
    /// deshalb wird der Mittelwert genommen. Danach werden sie zurückgesetzt.
    pub fn step(&self, model: &mut impl Trainable, batch_size: usize) {
        let scale = self.learning_rate / batch_size as f32;
        model.visit_parameters(&mut |parameter, gradient| *parameter -= scale * gradient);
        model.zero_grad();
    }
}
