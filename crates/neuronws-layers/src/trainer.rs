use crate::loss::Loss;
use crate::optimizer::{HasLearningRate, Optimizer};
use crate::scalar::Scalar;
use crate::schedule::Schedule;
use crate::training::{mean_loss, train_epoch};
use crate::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

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
    pub fn epoch<M, T, const IN: usize, const OUT: usize>(
        &mut self,
        model: &mut M,
        data: &[([T; IN], [T; OUT])],
    ) -> T
    where
        T: Scalar,
        M: ForwardPathMut<ForwardInput = [T; IN], ForwardOutput = [T; OUT]>
            + BackwardPathMut<BackwardInput = [T; OUT], BackwardOutput = [T; IN]>
            + Trainable<Scalar = T>,
        L: Loss<OUT>,
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

    /// `epochs` Runden. Liefert den Verlust der letzten Runde.
    ///
    /// # Panics
    /// Wenn `epochs` 0 ist oder `data` leer.
    pub fn fit<M, T, const IN: usize, const OUT: usize>(
        &mut self,
        model: &mut M,
        data: &[([T; IN], [T; OUT])],
        epochs: usize,
    ) -> T
    where
        T: Scalar,
        M: ForwardPathMut<ForwardInput = [T; IN], ForwardOutput = [T; OUT]>
            + BackwardPathMut<BackwardInput = [T; OUT], BackwardOutput = [T; IN]>
            + Trainable<Scalar = T>,
        L: Loss<OUT>,
        O: Optimizer<M>,
    {
        assert!(epochs > 0, "epochs muss größer als 0 sein");
        let mut last = T::ZERO;
        for _ in 0..epochs {
            last = self.epoch(model, data);
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
    pub fn fit_scheduled<M, S, T, const IN: usize, const OUT: usize>(
        &mut self,
        model: &mut M,
        data: &[([T; IN], [T; OUT])],
        epochs: usize,
        schedule: &S,
    ) -> T
    where
        T: Scalar,
        M: ForwardPathMut<ForwardInput = [T; IN], ForwardOutput = [T; OUT]>
            + BackwardPathMut<BackwardInput = [T; OUT], BackwardOutput = [T; IN]>
            + Trainable<Scalar = T>,
        L: Loss<OUT>,
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

    /// Mittlerer Verlust über alle Beispiele, ohne zu lernen.
    ///
    /// # Panics
    /// Wenn `data` leer ist.
    pub fn evaluate<M, T, const IN: usize, const OUT: usize>(
        &self,
        model: &M,
        data: &[([T; IN], [T; OUT])],
    ) -> T
    where
        T: Scalar,
        M: ForwardPath<ForwardInput = [T; IN], ForwardOutput = [T; OUT]>,
        L: Loss<OUT>,
    {
        mean_loss(model, &self.loss, data)
    }
}
