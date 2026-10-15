//! Das klassische Perceptron: ein Neuron, das mit einer Gerade zwischen zwei Klassen trennt.
//!
//! # Das Modell
//!
//! Ein Perceptron bildet aus den Eingaben `x` die gewichtete Summe plus Bias und fragt, ob sie
//! mindestens 0 ist:
//!
//! ```text
//! z = w · x + b            y = 1, wenn z ≥ 0, sonst 0
//! ```
//!
//! Geometrisch teilt `w · x + b = 0` den Raum der Eingaben in zwei Hälften: eine Gerade bei zwei
//! Eingaben, eine Ebene bei dreien. Auf der einen Seite sagt das Perceptron "1", auf der anderen
//! "0". `w` steht senkrecht auf der Gerade, `b` verschiebt sie.
//!
//! # Die klassische Lernregel
//!
//! Frank Rosenblatts Regel kommt ohne Ableitungen aus. Für jedes Beispiel `(x, t)` mit dem
//! gewünschten Ergebnis `t` (0 oder 1) rechnet das Perceptron zuerst seine Antwort `y` und ändert
//! dann seine Zahlen um einen Schritt `η` (die Lernrate):
//!
//! ```text
//! w ← w + η · (t − y) · x            b ← b + η · (t − y)
//! ```
//!
//! Der Fehler `t − y` kann nur drei Werte haben:
//!
//! | `t` | `y` | `t − y` | Wirkung |
//! |---|---|---|---|
//! | 1 | 1 | 0 | nichts, das Beispiel stimmt |
//! | 1 | 0 | +1 | `w` rückt auf `x` zu: Die Gerade wandert so, dass `x` eher auf die "1"-Seite fällt |
//! | 0 | 1 | −1 | `w` rückt von `x` weg: `x` fällt eher auf die "0"-Seite |
//!
//! Gelernt wird also **nur aus Fehlern**. Es gibt kein Maß dafür, "wie falsch" eine Antwort war,
//! nur ob sie falsch war.
//!
//! ## Konvergenz und ihre Grenze
//!
//! Gibt es eine Gerade, die die Beispiele trennt (die Daten sind *linear trennbar*), findet die
//! Regel nach endlich vielen Fehlern eine solche Gerade (Perceptron-Konvergenzsatz, Novikoff 1962).
//! Wie viele Fehler es höchstens werden, hängt vom Abstand der beiden Klassen ab (der "Marge"):
//! Liegen sie dicht beieinander, braucht es länger.
//!
//! Gibt es **keine** trennende Gerade, hört das Lernen nie auf, denn jede Gerade macht
//! mindestens einen Fehler und die Regel schiebt sie dann weiter. Das berühmteste Beispiel ist
//! XOR: Die Punkte `(0,0)` und `(1,1)` gehören zu "0", `(0,1)` und `(1,0)` zu "1", und keine
//! Gerade trennt sie. Marvin Minsky und Seymour Papert haben 1969 gezeigt, dass das ein
//! grundsätzliches Problem eines einzelnen Neurons ist. Der Ausweg sind mehrere Schichten: Ein
//! zweites Perceptron kann mit den Ausgaben der ersten rechnen, und zwei Geraden zusammen
//! (`OR` und `NAND`) schneiden das Rechteck aus, das XOR braucht. Das Beispiel `learn_gates`
//! führt es vor.
//!
//! ## Wie das mit dem Rest der Bibliothek zusammenpasst
//!
//! Die Regel ist **kein** Gradientenabstieg auf der Schwellenwertfunktion: Deren Ableitung ist
//! überall 0 (außer an der Sprungstelle), mit der Kettenregel käme nichts heraus. Sie ist aber
//! der Gradientenabstieg auf einer anderen Fehlerfunktion, dem "Perceptron-Kriterium" (nur
//! falsch eingeordnete Beispiele zählen, und zwar mit ihrem Abstand `|z|` zur Gerade). Setzt man
//! `y − t` als Gradienten an den Ausgang einer [`DenseLayer`] mit einem Neuron und lässt `Sgd`
//! einen Schritt machen, kommt genau dieselbe Änderung heraus. Der Test
//! `the_classical_rule_equals_one_gradient_step` in `tests/perceptron.rs` prüft das Bit für Bit.
//! Mit [`Perceptron::into_dense`] wird aus dem gelernten Perceptron eine Dense-Schicht, die man
//! mit anderen Schichten verketten und mit Gradienten weitertrainieren kann.
//!
//! Das Perceptron hält seine Zahlen selbst (statt eines `Block`s). Dadurch ist
//! [`Perceptron::new`] eine `const fn`, ein Perceptron lässt sich also zur Compile-Zeit anlegen und
//! in den Flash eines Mikrocontrollers legen. Ein `Block` kann das nicht, weil sein Konstruktor
//! über `Values::zeros()` ein Trait aufruft, und Trait-Methoden sind in `const fn` noch nicht
//! möglich.
//!
//! ```
//! use neuronws_perceptrons::Perceptron;
//!
//! // Ein Perceptron zur Compile-Zeit: die Gerade x + y = 1.5, also das logische UND.
//! const AND: Perceptron<2> = Perceptron::new([1.0, 1.0], -1.5);
//! assert_eq!(AND.predict(&[1.0, 1.0]), 1.0);
//! assert_eq!(AND.predict(&[1.0, 0.0]), 0.0);
//!
//! // Oder aus Beispielen lernen:
//! let mut learned = Perceptron::<2>::new([0.0, 0.0], 0.0);
//! let table = [([0.0, 0.0], 0.0), ([0.0, 1.0], 0.0), ([1.0, 0.0], 0.0), ([1.0, 1.0], 1.0)];
//! assert!(learned.fit(&table, 100, 0.1).is_some());
//! ```

#![no_std]

use neuronws_layers::activation::{Activation, Step};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::scalar::Scalar;
use neuronws_layers::traits::{ForwardData, ForwardPath, Layer};

/// Das klassische Perceptron: ein Neuron mit Schwellenwert-Aktivierung (`Step`).
///
/// Es ist `Copy` und kann in einer `const` angelegt werden. Weil es [`Layer`] und [`ForwardPath`]
/// implementiert, lässt es sich wie jede andere Schicht verketten.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Perceptron<const IN: usize, T: Scalar = f32> {
    weights: [T; IN],
    bias: T,
}

impl<const IN: usize, T: Scalar> Perceptron<IN, T> {
    /// Ein Perceptron mit gegebenen Gewichten und Bias. Eine `const fn`, siehe oben.
    pub const fn new(weights: [T; IN], bias: T) -> Self {
        Self { weights, bias }
    }

    pub const fn weights(&self) -> &[T; IN] {
        &self.weights
    }

    pub const fn bias(&self) -> T {
        self.bias
    }

    /// Die gewichtete Summe `z = w · x + b`, bevor der Schwellenwert angewandt wird. Ihr Vorzeichen
    /// sagt, auf welcher Seite der Gerade `x` liegt, ihr Betrag (geteilt durch die Länge von `w`)
    /// wie weit davon entfernt.
    pub fn net_input(&self, input: &[T; IN]) -> T {
        self.weights
            .iter()
            .zip(input)
            .fold(self.bias, |sum, (&w, &x)| sum + w * x)
    }

    /// 1, wenn `w · x + b ≥ 0`, sonst 0.
    pub fn predict(&self, input: &[T; IN]) -> T {
        Step.apply(self.net_input(input))
    }

    /// Ein Lernschritt mit der klassischen Regel `w ← w + η (t − y) x`, `b ← b + η (t − y)`.
    /// Liefert den Fehler `y − t`: 0, wenn richtig, sonst 1 oder −1.
    pub fn learn(&mut self, input: &[T; IN], target: T, learning_rate: T) -> T {
        let output = self.predict(input);
        let step = learning_rate * (target - output);
        for (weight, &x) in self.weights.iter_mut().zip(input) {
            *weight += step * x;
        }
        self.bias += step;
        output - target
    }

    /// Lernt aus den Beispielen, bis eine ganze Runde ohne Fehler durchläuft.
    ///
    /// Liefert die Anzahl der Runden, die es gebraucht hat (0, wenn schon die erste Runde fehlerfrei
    /// war). `None` heißt: Auch nach `max_epochs` Runden gab es noch Fehler. Das ist bei nicht
    /// linear trennbaren Daten (z. B. XOR) immer so.
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

    /// Wie viele der Beispiele das Perceptron falsch einordnet.
    pub fn mistakes(&self, data: &[([T; IN], T)]) -> usize {
        data.iter()
            .filter(|(input, target)| self.predict(input) != *target)
            .count()
    }

    /// Das Perceptron als [`DenseLayer`] mit einem Neuron (ohne die Schwellenwert-Aktivierung): Mit
    /// `Step` dahinter rechnet sie dasselbe. Damit lässt sich das Gelernte in ein größeres Netz
    /// einbauen oder mit Gradienten weitertrainieren.
    pub fn into_dense(self) -> DenseLayer<IN, 1, T> {
        DenseLayer::new([self.weights], [self.bias])
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
