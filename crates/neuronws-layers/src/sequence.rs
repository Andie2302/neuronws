//! Bausteine rund um Folgen: [`Embedding`], [`LastStep`] und [`MeanOverSteps`].
//!
//! Eine Folge hat in dieser Bibliothek die Form `[[T; D]; STEPS]`: `STEPS` Zeilen zu je `D`
//! Werten, `folge[zeitschritt][merkmal]`. Die rekurrenten Schichten in
//! [`recurrent`](crate::recurrent) verarbeiten sie. Die Bausteine hier bringen Text-artige
//! Eingaben hinein (`Embedding`) und machen aus einer Folge wieder eine einzige Zahlengruppe
//! (`LastStep`, `MeanOverSteps`).

use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;

/// Schlägt zu jeder Token-Nummer einen lernbaren Vektor nach.
///
/// Eingabe ist eine Folge von `STEPS` **Nummern** (zum Beispiel Wort-Nummern eines Wörterbuchs mit
/// `VOCAB` Einträgen), Ausgabe der Vektor `DIM` Zahlen zu jeder Nummer:
/// `y[t] = tabelle[nummer[t]]`. Ähnliche Tokens landen beim Lernen bei ähnlichen Vektoren.
///
/// Weil alles in dieser Bibliothek Zahlen vom Typ `T` sind, werden die Nummern **als Zahlen**
/// übergeben (`3.0` heißt Token 3). Bei `f32` sind das bis 16 777 216 genau. Nummern außerhalb von
/// `0..VOCAB` sind ein Fehler. Zwischenwerte wie `2.6` werden auf die nächste ganze Zahl (3)
/// gerundet.
///
/// Die Nummern selbst lassen sich nicht ableiten, ein Embedding ist nur über seine Tabelle
/// lernbar: Beim Zurückrechnen wird der Gradient jedes Schrittes zu der Tabellenzeile seines
/// Tokens addiert. Kommt ein Token mehrfach vor, summieren sich seine Beiträge. Der Gradient nach
/// den Nummern ist 0.
///
/// ```
/// use neuronws_layers::sequence::EmbeddingLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// // Wörterbuch mit 3 Tokens, jedes als Vektor aus 2 Zahlen
/// let embedding = EmbeddingLayer::<3, 2, 4>::new([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]]);
/// assert_eq!(
///     embedding.forward(&[2.0, 0.0, 2.0, 1.0]),
///     [[2.0, 2.1], [0.0, 0.1], [2.0, 2.1], [1.0, 1.1]]
/// );
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Embedding<const VOCAB: usize, const DIM: usize, const STEPS: usize, T: Scalar = f32>(
    PhantomData<T>,
);

impl<const VOCAB: usize, const DIM: usize, const STEPS: usize, T: Scalar>
    Embedding<VOCAB, DIM, STEPS, T>
{
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Die Token-Nummer, auf die eine Zahl zeigt (gerundet).
///
/// # Panics
/// Wenn die Nummer nicht in `0..vocab` liegt.
fn token<T: Scalar>(value: T, vocab: usize) -> usize {
    let rounded = value.to_f64() + 0.5;
    assert!(
        rounded >= 0.0 && rounded < vocab as f64,
        "Embedding: Token-Nummer {} liegt nicht in 0..{vocab}",
        value.to_f64()
    );
    rounded as usize
}

impl<const VOCAB: usize, const DIM: usize, const STEPS: usize, T: Scalar> Operation
    for Embedding<VOCAB, DIM, STEPS, T>
{
    type Scalar = T;
    type Input = [T; STEPS];
    type Output = [[T; DIM]; STEPS];
    /// Die Tabelle: eine Zeile pro Token.
    type Params = [[T; DIM]; VOCAB];

    fn forward(&self, table: &Self::Params, input: &Self::Input) -> Self::Output {
        core::array::from_fn(|t| table[token(input[t], VOCAB)])
    }

    fn backward(
        &self,
        _table: &Self::Params,
        grad_table: &mut Self::Params,
        input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        for (&id, grad) in input.iter().zip(grad_output) {
            let row = &mut grad_table[token(id, VOCAB)];
            for (target, &g) in row.iter_mut().zip(grad) {
                *target += g;
            }
        }
        [T::ZERO; STEPS]
    }

    /// Die Tabelle aus dem `Initializer`. Üblich sind kleine Zufallswerte (`Uniform`, `Normal`).
    fn init_params<I: Initializer>(&self, initializer: &I, rng: &mut Rng) -> Self::Params {
        let mut table = [[T::ZERO; DIM]; VOCAB];
        initializer.fill(rng, &mut table, DIM, VOCAB);
        table
    }
}

/// Eine Embedding-Schicht.
pub type EmbeddingLayer<const VOCAB: usize, const DIM: usize, const STEPS: usize, T = f32> =
    Block<Embedding<VOCAB, DIM, STEPS, T>>;

impl<const VOCAB: usize, const DIM: usize, const STEPS: usize, T: Scalar>
    Block<Embedding<VOCAB, DIM, STEPS, T>>
{
    pub fn new(table: [[T; DIM]; VOCAB]) -> Self {
        Self::from_parts(Embedding::new(), table)
    }

    pub const fn table(&self) -> &[[T; DIM]; VOCAB] {
        self.params()
    }

    pub const fn grad_table(&self) -> &[[T; DIM]; VOCAB] {
        self.grads()
    }
}

/// Nimmt aus einer Folge nur den **letzten** Schritt: `y = x[STEPS − 1]`.
///
/// Der letzte Zustand eines rekurrenten Netzes hat alle Schritte gesehen. Mit `LastStep` dahinter
/// wird aus "eine Ausgabe pro Schritt" eine einzige Ausgabe für die ganze Folge, etwa für die
/// Klassifikation. Rückwärts bekommt nur der letzte Schritt Gradient, alle anderen 0. Hat keine
/// Parameter.
///
/// ```
/// use neuronws_layers::sequence::LastStepLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// let layer = LastStepLayer::<3, 2>::new();
/// assert_eq!(layer.forward(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), [5.0, 6.0]);
/// ```
///
/// Eine Folge ohne Schritte hat keinen letzten Schritt. Das meldet der Compiler (erst beim
/// Bauen, siehe [`check_sizes!`](crate::check_sizes)):
///
/// ```compile_fail
/// use neuronws_layers::sequence::LastStep;
///
/// let _ = LastStep::<0, 2>::new();
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LastStep<const STEPS: usize, const D: usize, T: Scalar = f32>(PhantomData<T>);

impl<const STEPS: usize, const D: usize, T: Scalar> LastStep<STEPS, D, T> {
    /// # Compile-Fehler
    /// Wenn `STEPS` 0 ist (es gäbe keinen letzten Schritt).
    pub const fn new() -> Self {
        const {
            assert!(
                STEPS >= 1,
                "LastStep: die Folge braucht mindestens einen Schritt"
            )
        };
        Self(PhantomData)
    }
}

impl<const STEPS: usize, const D: usize, T: Scalar> Operation for LastStep<STEPS, D, T> {
    type Scalar = T;
    type Input = [[T; D]; STEPS];
    type Output = [T; D];
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        input[STEPS - 1]
    }

    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let mut grad_input = [[T::ZERO; D]; STEPS];
        grad_input[STEPS - 1] = *grad_output;
        grad_input
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Eine Schicht, die den letzten Schritt einer Folge herausgreift.
pub type LastStepLayer<const STEPS: usize, const D: usize, T = f32> = Block<LastStep<STEPS, D, T>>;

impl<const STEPS: usize, const D: usize, T: Scalar> Block<LastStep<STEPS, D, T>> {
    pub fn new() -> Self {
        Self::from_parts(LastStep::new(), ())
    }
}

impl<const STEPS: usize, const D: usize, T: Scalar> Default for Block<LastStep<STEPS, D, T>> {
    fn default() -> Self {
        Self::new()
    }
}

/// Mittelt eine Folge über die Zeit: `y = (x[0] + … + x[STEPS − 1]) / STEPS`.
///
/// Eine Alternative zu [`LastStep`], wenn alle Schritte gleich wichtig sind (zum Beispiel die
/// Stimmung eines Satzes aus allen seinen Wörtern). Rückwärts bekommt jeder Schritt den gleichen
/// Anteil `1/STEPS` des Gradienten. Hat keine Parameter.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanOverSteps<const STEPS: usize, const D: usize, T: Scalar = f32>(PhantomData<T>);

impl<const STEPS: usize, const D: usize, T: Scalar> MeanOverSteps<STEPS, D, T> {
    /// # Compile-Fehler
    /// Wenn `STEPS` 0 ist (es gäbe nichts zu mitteln).
    pub const fn new() -> Self {
        const {
            assert!(
                STEPS >= 1,
                "MeanOverSteps: die Folge braucht mindestens einen Schritt"
            )
        };
        Self(PhantomData)
    }
}

impl<const STEPS: usize, const D: usize, T: Scalar> Operation for MeanOverSteps<STEPS, D, T> {
    type Scalar = T;
    type Input = [[T; D]; STEPS];
    type Output = [T; D];
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        let steps = T::from_usize(STEPS);
        core::array::from_fn(|d| input.iter().fold(T::ZERO, |sum, row| sum + row[d]) / steps)
    }

    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let steps = T::from_usize(STEPS);
        [grad_output.map(|g| g / steps); STEPS]
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Eine Schicht, die eine Folge über die Zeit mittelt.
pub type MeanOverStepsLayer<const STEPS: usize, const D: usize, T = f32> =
    Block<MeanOverSteps<STEPS, D, T>>;

impl<const STEPS: usize, const D: usize, T: Scalar> Block<MeanOverSteps<STEPS, D, T>> {
    pub fn new() -> Self {
        Self::from_parts(MeanOverSteps::new(), ())
    }
}

impl<const STEPS: usize, const D: usize, T: Scalar> Default for Block<MeanOverSteps<STEPS, D, T>> {
    fn default() -> Self {
        Self::new()
    }
}
