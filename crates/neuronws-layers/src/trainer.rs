use crate::early_stopping::{EarlyStopping, Progress};
use crate::loss::Loss;
use crate::optimizer::{HasLearningRate, Optimizer};
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::schedule::{AdaptiveSchedule, Schedule};
use crate::training::{mean_loss, train_epoch};
use crate::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};
use crate::values::Values;

/// Ein Modell, das mit dem [`Trainer`] lernen kann: Es rechnet vorwärts und merkt sich dabei
/// Zwischenwerte (`ForwardPathMut`), rechnet Gradienten zurück (`BackwardPathMut`) und gibt seine
/// Parameter her (`Trainable`).
///
/// Das Trait fasst nur diese drei Fähigkeiten zusammen, damit die Signaturen unten lesbar bleiben.
/// Es wird für jedes passende Modell automatisch umgesetzt. Man implementiert es nie von Hand.
pub trait Learner<I, Out>:
    ForwardPathMut<ForwardInput = I, ForwardOutput = Out>
    + BackwardPathMut<BackwardInput = Out>
    + Trainable
{
}

impl<M, I, Out> Learner<I, Out> for M where
    M: ForwardPathMut<ForwardInput = I, ForwardOutput = Out>
        + BackwardPathMut<BackwardInput = Out>
        + Trainable
{
}

/// Das Ergebnis von [`Trainer::fit_validated`]: wie das Training verlaufen ist.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitReport<T> {
    /// Wie viele Runden gelernt wurden.
    pub epochs: usize,
    /// Die Runde mit dem besten Prüfverlust (die erste ist 0). Das Modell hat danach wieder genau
    /// die Parameter dieser Runde.
    pub best_epoch: usize,
    /// Der beste Prüfverlust.
    pub best_validation_loss: T,
    /// Der Verlust auf den Lerndaten in der letzten Runde.
    pub last_train_loss: T,
    /// `true`, wenn Early Stopping das Lernen vor `max_epochs` beendet hat.
    pub stopped_early: bool,
}

/// Bündelt, **wie** gelernt wird: Fehlerfunktion, Optimierer und Batchgröße.
///
/// Das Netz selbst gehört dem Aufrufer und wird bei jedem Aufruf übergeben. So lässt sich derselbe
/// Trainer für verschiedene Netze benutzen, und das Netz kennt keine Lernregel.
///
/// Einzelne Beispiele lernen heißt Batchgröße 1 (`Trainer::online`), alles auf einmal ist die
/// Länge der Daten.
///
/// ```
/// use neuronws_layers::activation::Identity;
/// use neuronws_layers::activation_layer::ActivationLayer;
/// use neuronws_layers::chain::LayerExt;
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::loss::MeanSquaredError;
/// use neuronws_layers::optimizer::Sgd;
/// use neuronws_layers::trainer::Trainer;
///
/// let mut net = DenseLayer::new([[0.0]], [0.0]).then(ActivationLayer::<1, _>::new(Identity));
/// let data = [([1.0], [3.0]), ([2.0], [5.0]), ([3.0], [7.0])];
///
/// let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 3);
/// let before = trainer.evaluate(&net, &data);
/// let after = trainer.fit(&mut net, &data, 200);
/// assert!(after < before);
/// ```
///
/// # Mit Prüfdaten und Early Stopping
///
/// Wer die Daten in Lern- und Prüfteil trennt ([`crate::data`]), lässt den Trainer nach jeder Runde
/// auf den Prüfdaten messen und aufhören, wenn es nicht mehr besser wird:
///
/// ```
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::early_stopping::EarlyStopping;
/// use neuronws_layers::loss::MeanSquaredError;
/// use neuronws_layers::optimizer::Sgd;
/// use neuronws_layers::trainer::Trainer;
///
/// let mut net = DenseLayer::new([[0.0]], [0.0]);
/// let train = [([1.0], [3.0]), ([2.0], [5.0]), ([3.0], [7.0])];
/// let validation = [([4.0], [9.0]), ([0.5], [2.0])];
///
/// let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 3);
/// let report = trainer.fit_validated(&mut net, &train, &validation, 1000, &mut EarlyStopping::new(5));
/// assert!(report.best_validation_loss < 0.01);
/// ```
#[derive(Debug, Clone)]
pub struct Trainer<L, O> {
    loss: L,
    optimizer: O,
    batch_size: usize,
}

impl<L, O> Trainer<L, O> {
    /// # Panics
    /// Wenn `batch_size` 0 ist.
    pub const fn new(loss: L, optimizer: O, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size muss größer als 0 sein");
        Self {
            loss,
            optimizer,
            batch_size,
        }
    }

    /// Ein Optimierungsschritt pro Beispiel (Batchgröße 1).
    pub const fn online(loss: L, optimizer: O) -> Self {
        Self::new(loss, optimizer, 1)
    }

    pub const fn loss(&self) -> &L {
        &self.loss
    }

    pub const fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub const fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Eine Runde über alle Beispiele. Liefert den mittleren Verlust der Runde (vor den Schritten
    /// gemessen, siehe [`train_epoch`]).
    ///
    /// # Panics
    /// Wenn `data` leer ist.
    pub fn epoch<M, T, I, Out>(&mut self, model: &mut M, data: &[(I, Out)]) -> T
    where
        T: Scalar,
        Out: Values<T>,
        M: Learner<I, Out> + Trainable<Scalar = T>,
        L: Loss,
        O: Optimizer<M>,
    {
        train_epoch(
            model,
            &mut self.optimizer,
            &self.loss,
            data,
            self.batch_size,
        )
    }

    /// Wie [`epoch`](Self::epoch), aber mischt die Beispiele vorher an Ort und Stelle.
    ///
    /// Bei jeder Runde eine andere Reihenfolge zu benutzen verhindert, dass das Netz sich an die
    /// Reihenfolge gewöhnt, und macht die Mini-Batches abwechslungsreicher. Weil das Mischen die
    /// Daten des Aufrufers umsortiert (ohne zusätzlichen Speicher), müssen sie veränderbar sein.
    ///
    /// # Panics
    /// Wenn `data` leer ist.
    pub fn epoch_shuffled<M, T, I, Out>(
        &mut self,
        model: &mut M,
        data: &mut [(I, Out)],
        rng: &mut Rng,
    ) -> T
    where
        T: Scalar,
        Out: Values<T>,
        M: Learner<I, Out> + Trainable<Scalar = T>,
        L: Loss,
        O: Optimizer<M>,
    {
        rng.shuffle(data);
        self.epoch(model, data)
    }

    /// `epochs` Runden. Liefert den Verlust der letzten Runde.
    ///
    /// # Panics
    /// Wenn `epochs` 0 ist oder `data` leer.
    pub fn fit<M, T, I, Out>(&mut self, model: &mut M, data: &[(I, Out)], epochs: usize) -> T
    where
        T: Scalar,
        Out: Values<T>,
        M: Learner<I, Out> + Trainable<Scalar = T>,
        L: Loss,
        O: Optimizer<M>,
    {
        assert!(epochs > 0, "epochs muss größer als 0 sein");
        let mut last = T::ZERO;
        for _ in 0..epochs {
            last = self.epoch(model, data);
        }
        last
    }

    /// Wie [`fit`](Self::fit), aber mit anderer Reihenfolge der Beispiele in jeder Runde, siehe
    /// [`epoch_shuffled`](Self::epoch_shuffled).
    ///
    /// # Panics
    /// Wenn `epochs` 0 ist oder `data` leer.
    pub fn fit_shuffled<M, T, I, Out>(
        &mut self,
        model: &mut M,
        data: &mut [(I, Out)],
        epochs: usize,
        rng: &mut Rng,
    ) -> T
    where
        T: Scalar,
        Out: Values<T>,
        M: Learner<I, Out> + Trainable<Scalar = T>,
        L: Loss,
        O: Optimizer<M>,
    {
        assert!(epochs > 0, "epochs muss größer als 0 sein");
        let mut last = T::ZERO;
        for _ in 0..epochs {
            last = self.epoch_shuffled(model, data, rng);
        }
        last
    }

    /// Wie [`fit`](Self::fit), aber die Lernrate folgt vor jeder Runde dem `schedule`.
    ///
    /// Als Start-Lernrate gilt die, die der Optimierer beim Aufruf hat. Nach dem Training bleibt
    /// die zuletzt gesetzte Lernrate im Optimierer.
    ///
    /// # Panics
    /// Wenn `epochs` 0 ist oder `data` leer.
    pub fn fit_scheduled<M, S, T, I, Out>(
        &mut self,
        model: &mut M,
        data: &[(I, Out)],
        epochs: usize,
        schedule: &S,
    ) -> T
    where
        T: Scalar,
        Out: Values<T>,
        M: Learner<I, Out> + Trainable<Scalar = T>,
        L: Loss,
        O: Optimizer<M> + HasLearningRate<T>,
        S: Schedule<T>,
    {
        assert!(epochs > 0, "epochs muss größer als 0 sein");
        let base = self.optimizer.learning_rate();
        let mut last = T::ZERO;
        for epoch in 0..epochs {
            self.optimizer.set_learning_rate(schedule.rate(epoch, base));
            last = self.epoch(model, data);
        }
        last
    }

    /// Wie [`fit`](Self::fit), aber ein Plan, der den **Verlust** sieht (zum Beispiel
    /// `ReduceOnPlateau`), bestimmt nach jeder Runde die nächste Lernrate.
    ///
    /// Der Plan bekommt den Verlust auf den Lerndaten. Will man auf den Prüfverlust reagieren,
    /// schreibt man die Schleife selbst: `epoch`, dann `evaluate`, dann
    /// `plateau.next_rate(verlust, trainer.optimizer().learning_rate())` und
    /// `trainer.optimizer_mut().set_learning_rate(...)`.
    ///
    /// # Panics
    /// Wenn `epochs` 0 ist oder `data` leer.
    pub fn fit_adaptive<M, S, T, I, Out>(
        &mut self,
        model: &mut M,
        data: &[(I, Out)],
        epochs: usize,
        schedule: &mut S,
    ) -> T
    where
        T: Scalar,
        Out: Values<T>,
        M: Learner<I, Out> + Trainable<Scalar = T>,
        L: Loss,
        O: Optimizer<M> + HasLearningRate<T>,
        S: AdaptiveSchedule<T>,
    {
        assert!(epochs > 0, "epochs muss größer als 0 sein");
        let mut last = T::ZERO;
        for _ in 0..epochs {
            last = self.epoch(model, data);
            let next = schedule.next_rate(last, self.optimizer.learning_rate());
            self.optimizer.set_learning_rate(next);
        }
        last
    }

    /// Lernt mit `train`, misst nach **jeder** Runde den Verlust auf `validation` und hört auf,
    /// wenn `stopping` meldet, dass er nicht mehr besser wird (oder nach `max_epochs`).
    ///
    /// Am Ende hat das Modell wieder die Parameter der **besten** Runde, nicht die der letzten
    /// (die schon überangepasst sein kann). Der Stand wird dafür als `M::Shape` gemerkt, also
    /// ohne `alloc`. Das Gedächtnis des Optimierers (Momentum, Adam) wird nicht zurückgesetzt.
    ///
    /// `stopping` sollte frisch sein (`EarlyStopping::new(geduld)`).
    ///
    /// # Panics
    /// Wenn `max_epochs` 0 ist oder `train` oder `validation` leer sind.
    pub fn fit_validated<M, T, I, Out>(
        &mut self,
        model: &mut M,
        train: &[(I, Out)],
        validation: &[(I, Out)],
        max_epochs: usize,
        stopping: &mut EarlyStopping<T>,
    ) -> FitReport<T>
    where
        T: Scalar,
        Out: Values<T>,
        M: Learner<I, Out>
            + ForwardPath<ForwardInput = I, ForwardOutput = Out>
            + Trainable<Scalar = T>,
        L: Loss,
        O: Optimizer<M>,
    {
        self.fit_validated_with(
            model,
            validation,
            max_epochs,
            stopping,
            &mut |trainer, model| trainer.epoch(model, train),
        )
    }

    /// Wie [`fit_validated`](Self::fit_validated), mischt aber die Lerndaten in jeder Runde an Ort
    /// und Stelle, siehe [`epoch_shuffled`](Self::epoch_shuffled).
    ///
    /// # Panics
    /// Wenn `max_epochs` 0 ist oder `train` oder `validation` leer sind.
    pub fn fit_validated_shuffled<M, T, I, Out>(
        &mut self,
        model: &mut M,
        train: &mut [(I, Out)],
        validation: &[(I, Out)],
        max_epochs: usize,
        stopping: &mut EarlyStopping<T>,
        rng: &mut Rng,
    ) -> FitReport<T>
    where
        T: Scalar,
        Out: Values<T>,
        M: Learner<I, Out>
            + ForwardPath<ForwardInput = I, ForwardOutput = Out>
            + Trainable<Scalar = T>,
        L: Loss,
        O: Optimizer<M>,
    {
        self.fit_validated_with(
            model,
            validation,
            max_epochs,
            stopping,
            &mut |trainer, model| trainer.epoch_shuffled(model, train, rng),
        )
    }

    /// Die gemeinsame Schleife von `fit_validated` und `fit_validated_shuffled`. `epoch` führt
    /// eine Lernrunde aus (mit oder ohne Mischen).
    fn fit_validated_with<M, T, I, Out>(
        &mut self,
        model: &mut M,
        validation: &[(I, Out)],
        max_epochs: usize,
        stopping: &mut EarlyStopping<T>,
        epoch: &mut dyn FnMut(&mut Self, &mut M) -> T,
    ) -> FitReport<T>
    where
        T: Scalar,
        Out: Values<T>,
        M: Learner<I, Out>
            + ForwardPath<ForwardInput = I, ForwardOutput = Out>
            + Trainable<Scalar = T>,
        L: Loss,
        O: Optimizer<M>,
    {
        assert!(max_epochs > 0, "max_epochs muss größer als 0 sein");
        let mut best = model.parameters();
        let mut report = FitReport {
            epochs: 0,
            best_epoch: 0,
            best_validation_loss: T::ZERO,
            last_train_loss: T::ZERO,
            stopped_early: false,
        };

        for round in 0..max_epochs {
            report.last_train_loss = epoch(self, model);
            report.epochs = round + 1;
            let validation_loss = self.evaluate(model, validation);
            match stopping.observe(validation_loss) {
                Progress::Improved => {
                    best = model.parameters();
                    report.best_epoch = round;
                    report.best_validation_loss = validation_loss;
                }
                Progress::Waiting => {}
                Progress::Stop => {
                    report.stopped_early = true;
                    break;
                }
            }
        }

        model.set_parameters(&best);
        report
    }

    /// Mittlerer Verlust über alle Beispiele, ohne zu lernen.
    ///
    /// # Panics
    /// Wenn `data` leer ist.
    pub fn evaluate<M, T, I, Out>(&self, model: &M, data: &[(I, Out)]) -> T
    where
        T: Scalar,
        Out: Values<T>,
        M: ForwardPath<ForwardInput = I, ForwardOutput = Out>,
        L: Loss,
    {
        mean_loss(model, &self.loss, data)
    }
}
