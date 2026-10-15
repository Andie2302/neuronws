#![no_std]

use neuronws_layers::activation::{Activation, Step};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::scalar::Scalar;
use neuronws_layers::traits::{BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer};

/// Das klassische Perceptron: ein Neuron mit Schwellenwert-Aktivierung.
///
/// Es ist eine `DenseLayer<IN, 1>` gefolgt von `Step`. Weil es `Layer` und `ForwardPath`
/// implementiert, lässt es sich wie jede andere Schicht verketten.
pub struct Perceptron<const IN: usize, T: Scalar = f32> {
    neuron: DenseLayer<IN, 1, T>,
}

impl<const IN: usize, T: Scalar> Perceptron<IN, T> {
    pub const fn new(weights: [T; IN], bias: T) -> Self {
        Self {
            neuron: DenseLayer::new([weights], [bias]),
        }
    }

    pub const fn weights(&self) -> &[T; IN] {
        &self.neuron.weights()[0]
    }

    pub const fn bias(&self) -> T {
        self.neuron.bias()[0]
    }

    /// 1, wenn `w · x + b ≥ 0`, sonst 0.
    pub fn predict(&self, input: &[T; IN]) -> T {
        let [z] = self.neuron.forward(input);
        Step.apply(z)
    }

    /// Ein Lernschritt mit der Perceptron-Regel `w ← w + η (t − y) x`, `b ← b + η (t − y)`.
    /// Liefert den Fehler `y − t`: 0, wenn richtig, sonst 1 oder −1.
    ///
    /// Die Regel braucht keinen eigenen Code. Mit dem Gradienten `y − t` am Ausgang rechnet der
    /// Backward-Pass der Dense-Schicht `∂L/∂w = (y − t) · x`, und `Sgd` zieht das ab.
    pub fn learn(&mut self, input: &[T; IN], target: T, learning_rate: T) -> T {
        let error = self.predict(input) - target;
        self.neuron.forward_mut(input);
        self.neuron.backward_mut(&[error]);
        Sgd::new(learning_rate).step(&mut self.neuron, 1);
        error
    }

    /// Lernt aus den Beispielen, bis eine ganze Runde ohne Fehler durchläuft.
    ///
    /// Liefert die Anzahl der Runden, die es gebraucht hat. `None` heißt: Auch nach `max_epochs`
    /// Runden gab es noch Fehler. Das ist bei nicht linear trennbaren Daten (z. B. XOR) immer so.
    pub fn fit(
        &mut self,
        data: &[([T; IN], T)],
        max_epochs: usize,
        learning_rate: T,
    ) -> Option<usize> {
        for epoch in 0..max_epochs {
            let mut mistakes = 0;
            for (input, target) in data {
                if self.learn(input, *target, learning_rate) != T::ZERO {
                    mistakes += 1;
                }
            }
            if mistakes == 0 {
                return Some(epoch);
            }
        }
        None
    }
}

impl<const IN: usize, T: Scalar> Layer for Perceptron<IN, T> {}

impl<const IN: usize, T: Scalar> ForwardData for Perceptron<IN, T> {
    type ForwardInput = [T; IN];
    type ForwardOutput = [T; 1];
}

impl<const IN: usize, T: Scalar> ForwardPath for Perceptron<IN, T> {
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        [self.predict(input)]
    }
}
