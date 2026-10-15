//! Optimierer mit einer eigenen Schrittweite pro Parameter: [`RmsProp`] und [`AdaGrad`].
//!
//! Beide merken sich pro Parameter, wie groß seine Gradienten bisher waren, und teilen durch die
//! Wurzel davon. Parameter mit großen Gradienten machen so kleinere Schritte, Parameter mit
//! kleinen größere. Das gleicht schlecht skalierte Aufgaben aus. [`Adam`](crate::adam::Adam) baut
//! auf derselben Idee auf und kommt noch mit einem Impuls dazu.
//!
//! Beide brauchen die Wurzel, deshalb gibt es sie nur mit dem Feature `std` oder `libm`.

use core::fmt;

use crate::optimizer::{HasLearningRate, Optimizer, batch_divisor};
use crate::scalar::Scalar;
use crate::traits::Trainable;
use crate::values::Values;

/// RMSprop: `s ← ρ s + (1 − ρ) g²`, `p ← p − η g / (√s + ε)`.
///
/// `s` ist ein gleitender Mittelwert der quadrierten Gradienten: Alte Werte verblassen mit dem
/// Faktor `ρ` (Standard 0,9). Anders als bei [`AdaGrad`] kann `s` deshalb auch wieder sinken, und
/// die Schritte werden nicht dauerhaft immer kleiner. `ε` (Standard 1e-8) verhindert die Division
/// durch 0.
///
/// ```
/// use neuronws_layers::adaptive::RmsProp;
/// use neuronws_layers::dense_layer::DenseLayer;
///
/// let layer = DenseLayer::new([[0.0; 2]], [0.0]);
/// let _rmsprop = RmsProp::for_model(&layer, 0.01);
/// ```
pub struct RmsProp<M: Trainable> {
    learning_rate: M::Scalar,
    decay: M::Scalar,
    epsilon: M::Scalar,
    mean_square: M::Shape,
}

impl<M: Trainable> RmsProp<M> {
    /// Mit den üblichen Werten `ρ = 0.9` und `ε = 1e-8`.
    pub fn new(learning_rate: M::Scalar) -> Self {
        Self {
            learning_rate,
            decay: M::Scalar::from_f32(0.9),
            epsilon: M::Scalar::from_f32(1e-8),
            mean_square: <M::Shape as Values<M::Scalar>>::zeros(),
        }
    }

    /// Wie `new`, aber der Typ des Netzes ergibt sich aus einem vorhandenen Netz.
    pub fn for_model(_model: &M, learning_rate: M::Scalar) -> Self {
        Self::new(learning_rate)
    }

    /// Eigene Werte für `ρ` und `ε`.
    #[must_use]
    pub const fn with_decay(mut self, decay: M::Scalar, epsilon: M::Scalar) -> Self {
        self.decay = decay;
        self.epsilon = epsilon;
        self
    }
}

impl<M: Trainable> HasLearningRate<M::Scalar> for RmsProp<M> {
    fn learning_rate(&self) -> M::Scalar {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: M::Scalar) {
        self.learning_rate = learning_rate;
    }
}

impl<M: Trainable> Optimizer<M> for RmsProp<M> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        let batch = batch_divisor::<M::Scalar>(batch_size);
        let (learning_rate, decay, epsilon) = (self.learning_rate, self.decay, self.epsilon);
        let one = M::Scalar::ONE;
        let mean_square = &mut self.mean_square;
        let mut index = 0;

        model.visit_parameters(&mut |parameter, gradient| {
            let g = *gradient / batch;
            let s = mean_square.value_mut(index);
            *s = decay * *s + (one - decay) * g * g;
            *parameter -= learning_rate * g / (s.sqrt() + epsilon);
            index += 1;
        });

        model.zero_grad();
    }
}

impl<M: Trainable> Clone for RmsProp<M> {
    fn clone(&self) -> Self {
        Self {
            learning_rate: self.learning_rate,
            decay: self.decay,
            epsilon: self.epsilon,
            mean_square: self.mean_square,
        }
    }
}

impl<M: Trainable> fmt::Debug for RmsProp<M>
where
    M::Scalar: fmt::Debug,
    M::Shape: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RmsProp")
            .field("learning_rate", &self.learning_rate)
            .field("decay", &self.decay)
            .field("epsilon", &self.epsilon)
            .field("mean_square", &self.mean_square)
            .finish()
    }
}

/// AdaGrad: `s ← s + g²`, `p ← p − η g / (√s + ε)`.
///
/// `s` summiert alle bisherigen quadrierten Gradienten auf und wächst nur. Parameter, die oft
/// große Gradienten sehen, werden dadurch schnell vorsichtig. Selten angesprochene Parameter
/// (zum Beispiel ein Wort, das nur selten vorkommt) behalten große Schritte. Der Preis: Die
/// Schritte schrumpfen im Lauf des Trainings immer weiter, irgendwann kommt kaum noch etwas
/// voran. `ε` (Standard 1e-8) verhindert die Division durch 0.
///
/// ```
/// use neuronws_layers::adaptive::AdaGrad;
/// use neuronws_layers::dense_layer::DenseLayer;
///
/// let layer = DenseLayer::new([[0.0; 2]], [0.0]);
/// let _adagrad = AdaGrad::for_model(&layer, 0.1);
/// ```
pub struct AdaGrad<M: Trainable> {
    learning_rate: M::Scalar,
    epsilon: M::Scalar,
    sum_of_squares: M::Shape,
}

impl<M: Trainable> AdaGrad<M> {
    /// Mit `ε = 1e-8`.
    pub fn new(learning_rate: M::Scalar) -> Self {
        Self {
            learning_rate,
            epsilon: M::Scalar::from_f32(1e-8),
            sum_of_squares: <M::Shape as Values<M::Scalar>>::zeros(),
        }
    }

    /// Wie `new`, aber der Typ des Netzes ergibt sich aus einem vorhandenen Netz.
    pub fn for_model(_model: &M, learning_rate: M::Scalar) -> Self {
        Self::new(learning_rate)
    }

    /// Ein eigenes `ε`.
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: M::Scalar) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl<M: Trainable> HasLearningRate<M::Scalar> for AdaGrad<M> {
    fn learning_rate(&self) -> M::Scalar {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: M::Scalar) {
        self.learning_rate = learning_rate;
    }
}

impl<M: Trainable> Optimizer<M> for AdaGrad<M> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        let batch = batch_divisor::<M::Scalar>(batch_size);
        let (learning_rate, epsilon) = (self.learning_rate, self.epsilon);
        let sum_of_squares = &mut self.sum_of_squares;
        let mut index = 0;

        model.visit_parameters(&mut |parameter, gradient| {
            let g = *gradient / batch;
            let s = sum_of_squares.value_mut(index);
            *s += g * g;
            *parameter -= learning_rate * g / (s.sqrt() + epsilon);
            index += 1;
        });

        model.zero_grad();
    }
}

impl<M: Trainable> Clone for AdaGrad<M> {
    fn clone(&self) -> Self {
        Self {
            learning_rate: self.learning_rate,
            epsilon: self.epsilon,
            sum_of_squares: self.sum_of_squares,
        }
    }
}

impl<M: Trainable> fmt::Debug for AdaGrad<M>
where
    M::Scalar: fmt::Debug,
    M::Shape: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdaGrad")
            .field("learning_rate", &self.learning_rate)
            .field("epsilon", &self.epsilon)
            .field("sum_of_squares", &self.sum_of_squares)
            .finish()
    }
}
