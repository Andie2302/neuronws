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
    /// Ein Lernschritt aus den gesammelten Gradienten von `batch_size` Beispielen.
    ///
    /// # Panics
    /// Wenn `batch_size` 0 ist. Es gäbe nichts zu mitteln, und die Division durch 0 würde alle
    /// Parameter stillschweigend zu `NaN` machen. Ein eigener Fehlertyp wäre hier übertrieben: Die
    /// Batchgröße kommt in der Praxis aus `chunks()` oder ist eine Konstante, eine 0 ist ein
    /// Programmierfehler und kein Zustand, auf den man zur Laufzeit reagieren kann.
    fn step(&mut self, model: &mut M, batch_size: usize);
}

/// Die Batchgröße als Zahl, zum Mitteln der aufsummierten Gradienten.
///
/// # Panics
/// Wenn `batch_size` 0 ist, siehe [`Optimizer::step`].
pub(crate) fn batch_divisor<T: Scalar>(batch_size: usize) -> T {
    assert!(batch_size > 0, "batch_size muss größer als 0 sein");
    T::from_usize(batch_size)
}

/// Ein Optimierer mit einer Lernrate `η`, die man lesen und ändern kann.
///
/// Das braucht ein Lernraten-Plan ([`crate::schedule`]), der `η` im Lauf des Trainings anpasst.
pub trait HasLearningRate<T: Scalar> {
    fn learning_rate(&self) -> T;
    fn set_learning_rate(&mut self, learning_rate: T);
}

/// Stochastic Gradient Descent: `p ← p − η · g`.
///
/// # Stolperstein: `f32` oder `f64`?
///
/// Der Zahlentyp von `Sgd<T>` muss zum Netz passen. Bei einem Literal wie `Sgd::new(0.1)` leitet
/// Rust ihn meist aus der späteren Benutzung ab (das Netz rechnet mit `f32`, also ist es `f32`).
/// Gibt es **nichts**, woraus Rust ihn ableiten kann, nimmt es für eine Kommazahl-Konstante `f64`.
/// Und eine Variable hat ihren Typ schon: Kommt der Wert aus einer `f64`-Rechnung, wird es
/// `Sgd<f64>`, und der Fehler (`Sgd<f64>` ist kein Optimierer für dieses Netz) erscheint erst
/// bei `step`, weit weg von der Ursache:
///
/// ```compile_fail
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::optimizer::{Optimizer, Sgd};
///
/// let learning_rate: f64 = 0.1; // zum Beispiel das Ergebnis einer Rechnung mit f64
/// let mut sgd = Sgd::new(learning_rate); // also `Sgd<f64>`
/// let mut net = DenseLayer::new([[0.0_f32]], [0.0]);
/// sgd.step(&mut net, 1); // Fehler: `Sgd<f64>` passt nicht zu einem `f32`-Netz
/// ```
///
/// Es gibt drei Wege, das zu vermeiden. Am einfachsten ist [`Sgd::for_model`], dort bestimmt das
/// Netz den Typ:
///
/// ```
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::optimizer::{Optimizer, Sgd};
///
/// let mut net = DenseLayer::new([[0.0_f32]], [0.0]);
///
/// let mut by_model = Sgd::for_model(&net, 0.1); // der Typ folgt aus dem Netz
/// let mut by_suffix = Sgd::new(0.1_f32); // oder ausdrücklich über das Suffix
/// let mut by_turbofish = Sgd::<f32>::new(0.1); // oder über den Typ
/// by_model.step(&mut net, 1);
/// by_suffix.step(&mut net, 1);
/// by_turbofish.step(&mut net, 1);
/// ```
#[derive(Debug, Clone)]
pub struct Sgd<T: Scalar = f32> {
    learning_rate: T,
}

impl<T: Scalar> Sgd<T> {
    pub const fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }

    /// Wie `new`, aber der Zahlentyp ergibt sich aus dem Netz: Ein `f32`-Netz bekommt einen
    /// `Sgd<f32>`, ein `f64`-Netz einen `Sgd<f64>`, auch wenn die Lernrate ein nacktes Literal ist.
    /// Das vermeidet den Stolperstein oben. (`Momentum`, `RmsProp` und `Adam` haben dasselbe.)
    pub fn for_model<M: Trainable<Scalar = T>>(_model: &M, learning_rate: T) -> Self {
        Self::new(learning_rate)
    }
}

impl<T: Scalar> HasLearningRate<T> for Sgd<T> {
    fn learning_rate(&self) -> T {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: T) {
        self.learning_rate = learning_rate;
    }
}

impl<M: Trainable> Optimizer<M> for Sgd<M::Scalar> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        let scale = self.learning_rate / batch_divisor::<M::Scalar>(batch_size);
        model.visit_parameters(&mut |parameter, gradient| *parameter -= scale * *gradient);
        model.zero_grad();
    }
}

/// SGD mit Impuls: `v ← μ v − η g`, `p ← p + v`.
///
/// Jeder Parameter merkt sich seine bisherige "Geschwindigkeit" `v`. Das glättet Zickzack-Wege
/// und beschleunigt das Lernen in Richtungen, in denen der Gradient immer gleich zeigt.
///
/// Mit [`with_nesterov`](Momentum::with_nesterov) wird daraus **Nesterov-Momentum**: Der
/// Parameter springt nicht nur um die neue Geschwindigkeit, sondern schaut dabei schon einen
/// Impuls-Schritt voraus: `p ← p + μ v − η g` (mit dem neuen `v`). Bei gekrümmten Tälern bremst
/// das früher ab und schwingt weniger über das Ziel hinaus.
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
/// let _nesterov = Momentum::for_model(&net, 0.01, 0.9).with_nesterov();
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
    nesterov: bool,
    velocity: M::Shape,
}

impl<M: Trainable> Momentum<M> {
    pub fn new(learning_rate: M::Scalar, momentum: M::Scalar) -> Self {
        Self {
            learning_rate,
            momentum,
            nesterov: false,
            velocity: <M::Shape as Values<M::Scalar>>::zeros(),
        }
    }

    /// ... oder über ein vorhandenes Netz, ohne den Typ hinzuschreiben.
    pub fn for_model(_model: &M, learning_rate: M::Scalar, momentum: M::Scalar) -> Self {
        Self::new(learning_rate, momentum)
    }

    /// Schaltet auf Nesterov-Momentum um, siehe oben.
    #[must_use]
    pub const fn with_nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }

    pub const fn is_nesterov(&self) -> bool {
        self.nesterov
    }
}

impl<M: Trainable> HasLearningRate<M::Scalar> for Momentum<M> {
    fn learning_rate(&self) -> M::Scalar {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: M::Scalar) {
        self.learning_rate = learning_rate;
    }
}

impl<M: Trainable> Optimizer<M> for Momentum<M> {
    fn step(&mut self, model: &mut M, batch_size: usize) {
        let scale = self.learning_rate / batch_divisor::<M::Scalar>(batch_size);
        let momentum = self.momentum;
        let nesterov = self.nesterov;
        let velocity = &mut self.velocity;
        let mut index = 0;

        model.visit_parameters(&mut |parameter, gradient| {
            let step = scale * *gradient;
            let v = velocity.value_mut(index);
            *v = momentum * *v - step;
            *parameter += if nesterov { momentum * *v - step } else { *v };
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
            nesterov: self.nesterov,
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
            .field("nesterov", &self.nesterov)
            .field("velocity", &self.velocity)
            .finish()
    }
}
