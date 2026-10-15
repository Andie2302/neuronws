use core::fmt;

use crate::optimizer::{HasLearningRate, Optimizer, batch_divisor};
use crate::scalar::Scalar;
use crate::traits::Trainable;
use crate::values::Values;

/// AdamW: Adam mit entkoppeltem Weight Decay.
///
/// Jeder Parameter bekommt seine eigene Schrittweite. Dafür merkt sich der Optimierer pro
/// Parameter zwei Werte: `m`, den gleitenden Mittelwert des Gradienten (die Richtung), und `v`,
/// den gleitenden Mittelwert des quadrierten Gradienten (wie stark er schwankt).
///
/// ```text
/// m ← β₁ m + (1 − β₁) g          m̂ = m / (1 − β₁ᵗ)
/// v ← β₂ v + (1 − β₂) g²         v̂ = v / (1 − β₂ᵗ)
/// p ← p − η · ( m̂ / (√v̂ + ε) + λ p )
/// ```
///
/// `m̂` und `v̂` korrigieren, dass `m` und `v` am Anfang bei 0 starten und deshalb zu klein sind.
/// `λ` (`weight_decay`) zieht die Gewichte leicht Richtung 0. Bei AdamW passiert das **neben**
/// dem Adam-Schritt und nicht über den Gradienten, deshalb "entkoppelt".
///
/// Braucht die Wurzel, deshalb nur mit dem Feature `std` oder `libm`.
pub struct AdamW<M: Trainable> {
    learning_rate: M::Scalar,
    beta1: M::Scalar,
    beta2: M::Scalar,
    epsilon: M::Scalar,
    weight_decay: M::Scalar,
    /// `β₁ᵗ` und `β₂ᵗ`, werden pro Schritt mit `β` multipliziert.
    beta1_power: M::Scalar,
    beta2_power: M::Scalar,
    m: M::Shape,
    v: M::Shape,
}

impl<M: Trainable> AdamW<M> {
    /// Mit den üblichen Werten `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.
    pub fn new(learning_rate: M::Scalar, weight_decay: M::Scalar) -> Self {
        Self {
            learning_rate,
            beta1: M::Scalar::from_f32(0.9),
            beta2: M::Scalar::from_f32(0.999),
            epsilon: M::Scalar::from_f32(1e-8),
            weight_decay,
            beta1_power: M::Scalar::ONE,
            beta2_power: M::Scalar::ONE,
            m: <M::Shape as Values<M::Scalar>>::zeros(),
            v: <M::Shape as Values<M::Scalar>>::zeros(),
        }
    }

    /// Wie `new`, aber der Typ des Netzes ergibt sich aus einem vorhandenen Netz.
    pub fn for_model(_model: &M, learning_rate: M::Scalar, weight_decay: M::Scalar) -> Self {
        Self::new(learning_rate, weight_decay)
    }

    /// Eigene Werte für `β₁`, `β₂` und `ε`.
    #[must_use]
    pub fn with_betas(mut self, beta1: M::Scalar, beta2: M::Scalar, epsilon: M::Scalar) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.epsilon = epsilon;
        self
    }
}

/// Die Lernrate darf sich zwischen den Schritten ändern (siehe [`crate::schedule`]).
impl<M: Trainable> HasLearningRate<M::Scalar> for AdamW<M> {
    fn learning_rate(&self) -> M::Scalar {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: M::Scalar) {
        self.learning_rate = learning_rate;
    }
}

impl<M: Trainable> Optimizer<M> for AdamW<M> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        let one = M::Scalar::ONE;
        let batch = batch_divisor::<M::Scalar>(batch_size);
        self.beta1_power = self.beta1_power * self.beta1;
        self.beta2_power = self.beta2_power * self.beta2;
        let correction1 = one - self.beta1_power;
        let correction2 = one - self.beta2_power;

        let (beta1, beta2) = (self.beta1, self.beta2);
        let (learning_rate, epsilon, weight_decay) =
            (self.learning_rate, self.epsilon, self.weight_decay);
        let (m, v) = (&mut self.m, &mut self.v);
        let mut index = 0;

        model.visit_parameters(&mut |parameter, gradient| {
            let g = *gradient / batch;
            let m = m.value_mut(index);
            let v = v.value_mut(index);
            *m = beta1 * *m + (one - beta1) * g;
            *v = beta2 * *v + (one - beta2) * g * g;
            let m_hat = *m / correction1;
            let v_hat = *v / correction2;
            let adam_step = m_hat / (v_hat.sqrt() + epsilon);
            *parameter -= learning_rate * (adam_step + weight_decay * *parameter);
            index += 1;
        });

        model.zero_grad();
    }
}

impl<M: Trainable> Clone for AdamW<M> {
    fn clone(&self) -> Self {
        Self {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
            beta1_power: self.beta1_power,
            beta2_power: self.beta2_power,
            m: self.m,
            v: self.v,
        }
    }
}

impl<M: Trainable> fmt::Debug for AdamW<M>
where
    M::Scalar: fmt::Debug,
    M::Shape: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdamW")
            .field("learning_rate", &self.learning_rate)
            .field("weight_decay", &self.weight_decay)
            .field("m", &self.m)
            .field("v", &self.v)
            .finish_non_exhaustive()
    }
}

/// Adam: wie [`AdamW`], aber ohne Weight Decay (`λ = 0`).
///
/// ```
/// use neuronws_layers::adam::Adam;
/// use neuronws_layers::dense_layer::DenseLayer;
///
/// let layer = DenseLayer::new([[0.0; 2]], [0.0]);
/// let _adam = Adam::for_model(&layer, 0.001);
/// ```
pub struct Adam<M: Trainable>(AdamW<M>);

impl<M: Trainable> Adam<M> {
    pub fn new(learning_rate: M::Scalar) -> Self {
        Self(AdamW::new(learning_rate, M::Scalar::ZERO))
    }

    pub fn for_model(_model: &M, learning_rate: M::Scalar) -> Self {
        Self::new(learning_rate)
    }

    #[must_use]
    pub fn with_betas(self, beta1: M::Scalar, beta2: M::Scalar, epsilon: M::Scalar) -> Self {
        Self(self.0.with_betas(beta1, beta2, epsilon))
    }
}

impl<M: Trainable> HasLearningRate<M::Scalar> for Adam<M> {
    fn learning_rate(&self) -> M::Scalar {
        self.0.learning_rate()
    }

    fn set_learning_rate(&mut self, learning_rate: M::Scalar) {
        self.0.set_learning_rate(learning_rate);
    }
}

impl<M: Trainable> Optimizer<M> for Adam<M> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        self.0.step(model, batch_size);
    }
}

impl<M: Trainable> Clone for Adam<M> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<M: Trainable> fmt::Debug for Adam<M>
where
    M::Scalar: fmt::Debug,
    M::Shape: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Adam").field(&self.0).finish()
    }
}
