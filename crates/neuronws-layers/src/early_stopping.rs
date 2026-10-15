//! Early Stopping: das Lernen abbrechen, wenn der Verlust auf den **Prüfdaten** wieder steigt.
//!
//! Der Verlust auf den Lerndaten sinkt fast immer weiter. Der auf Daten, die das Netz nicht zum
//! Lernen benutzt (Validierungsdaten), sinkt zuerst, erreicht ein Minimum und steigt dann wieder:
//! Das Netz beginnt, Rauschen in den Lerndaten auswendig zu lernen (Overfitting). Early Stopping
//! beobachtet den Prüfverlust und hört auf, sobald er `patience` Runden lang nicht mehr besser
//! wurde. Benutzt wird es meistens über `Trainer::fit_validated`, das auch das beste Modell
//! wiederherstellt.

use crate::scalar::Scalar;

/// Was ein neuer Prüfverlust für das Training bedeutet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// Der beste Wert bisher. Jetzt lohnt es sich, den Stand des Netzes zu merken.
    Improved,
    /// Nicht besser als der beste Wert, aber die Geduld reicht noch.
    Waiting,
    /// Die Geduld ist aufgebraucht: aufhören.
    Stop,
}

/// Beobachtet den Prüfverlust von Runde zu Runde und sagt, wann Schluss ist.
///
/// ```
/// use neuronws_layers::early_stopping::{EarlyStopping, Progress};
///
/// // Geduld 2: Nach der zweiten Runde in Folge ohne Verbesserung ist Schluss.
/// let mut stopping = EarlyStopping::new(2);
/// let verdicts = [0.9, 0.7, 0.8, 0.75].map(|loss| stopping.observe(loss));
/// assert_eq!(
///     verdicts,
///     [Progress::Improved, Progress::Improved, Progress::Waiting, Progress::Stop]
/// );
/// assert_eq!(stopping.best(), Some(0.7));
/// assert_eq!(stopping.best_epoch(), 1); // Runde 1 (die zweite) war die beste
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EarlyStopping<T: Scalar = f32> {
    patience: usize,
    min_delta: T,
    best: Option<T>,
    best_epoch: usize,
    epochs_seen: usize,
    rounds_without_progress: usize,
}

impl<T: Scalar> EarlyStopping<T> {
    /// Hört auf, wenn `patience` Runden hintereinander keine Verbesserung kamen.
    pub const fn new(patience: usize) -> Self {
        Self {
            patience,
            min_delta: T::ZERO,
            best: None,
            best_epoch: 0,
            epochs_seen: 0,
            rounds_without_progress: 0,
        }
    }

    /// Eine Verbesserung zählt erst, wenn sie mindestens so groß ist (Standard 0). Verhindert, dass
    /// winzige Schwankungen das Zählen ständig zurücksetzen.
    #[must_use]
    pub const fn with_min_delta(mut self, min_delta: T) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// Meldet den Prüfverlust einer Runde.
    pub fn observe(&mut self, validation_loss: T) -> Progress {
        let epoch = self.epochs_seen;
        self.epochs_seen += 1;

        let improved = self
            .best
            .is_none_or(|best| validation_loss < best - self.min_delta);
        if improved {
            self.best = Some(validation_loss);
            self.best_epoch = epoch;
            self.rounds_without_progress = 0;
            return Progress::Improved;
        }

        self.rounds_without_progress += 1;
        if self.rounds_without_progress >= self.patience {
            Progress::Stop
        } else {
            Progress::Waiting
        }
    }

    /// Der beste Prüfverlust bisher.
    pub const fn best(&self) -> Option<T> {
        self.best
    }

    /// Die Nummer der besten Runde (die erste ist 0).
    pub const fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// Wie viele Runden gemeldet wurden.
    pub const fn epochs_seen(&self) -> usize {
        self.epochs_seen
    }

    /// Wie viele Runden in Folge es keine Verbesserung gab.
    pub const fn rounds_without_progress(&self) -> usize {
        self.rounds_without_progress
    }
}
