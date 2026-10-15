//! Lernraten-Pläne: wie sich die Lernrate `η` im Lauf des Trainings ändert.
//!
//! Am Anfang helfen große Schritte, am Ende kleine, damit das Netz nicht um das Minimum herum
//! springt. Ein Plan sagt nur, **wie viel von der Start-Lernrate** in Runde `epoch` gelten soll.

use crate::scalar::Scalar;

/// Berechnet die Lernrate für eine Runde aus der Start-Lernrate `base`.
///
/// Eigene Pläne sind ein einzelnes `impl`, genau wie bei `Activation` oder `Loss`.
pub trait Schedule<T: Scalar> {
    /// Die Lernrate für Runde `epoch` (0 ist die erste).
    fn rate(&self, epoch: usize, base: T) -> T;
}

/// Die Lernrate bleibt, wie sie ist.
#[derive(Debug, Clone, Copy, Default)]
pub struct Constant;

impl<T: Scalar> Schedule<T> for Constant {
    fn rate(&self, _epoch: usize, base: T) -> T {
        base
    }
}

/// Alle `every` Runden wird die Lernrate mit `factor` multipliziert: `η = base · factor^(epoch / every)`.
///
/// `StepDecay::new(0.5, 100)` halbiert die Lernrate alle 100 Runden.
#[derive(Debug, Clone, Copy)]
pub struct StepDecay<T: Scalar = f32> {
    factor: T,
    every: usize,
}

impl<T: Scalar> StepDecay<T> {
    /// # Panics
    /// Wenn `every` 0 ist.
    pub const fn new(factor: T, every: usize) -> Self {
        assert!(every > 0, "every muss größer als 0 sein");
        Self { factor, every }
    }
}

impl<T: Scalar> Schedule<T> for StepDecay<T> {
    fn rate(&self, epoch: usize, base: T) -> T {
        (0..epoch / self.every).fold(base, |rate, _| rate * self.factor)
    }
}

/// Die Lernrate sinkt gleichmäßig von `base` auf `base · final_fraction` in `epochs` Runden und
/// bleibt danach dort.
#[derive(Debug, Clone, Copy)]
pub struct LinearDecay<T: Scalar = f32> {
    final_fraction: T,
    epochs: usize,
}

impl<T: Scalar> LinearDecay<T> {
    /// # Panics
    /// Wenn `epochs` 0 ist.
    pub const fn new(final_fraction: T, epochs: usize) -> Self {
        assert!(epochs > 0, "epochs muss größer als 0 sein");
        Self {
            final_fraction,
            epochs,
        }
    }
}

impl<T: Scalar> Schedule<T> for LinearDecay<T> {
    fn rate(&self, epoch: usize, base: T) -> T {
        let progress = T::from_usize(epoch.min(self.epochs)) / T::from_usize(self.epochs);
        base * (T::ONE - progress * (T::ONE - self.final_fraction))
    }
}

/// Die Lernrate sinkt entlang einer halben Cosinus-Kurve von `base` auf `base · final_fraction`
/// in `epochs` Runden und bleibt danach dort.
///
/// ```text
/// η = base · ( f + (1 − f) · ½ · (1 + cos(π · Runde / epochs)) )     mit f = final_fraction
/// ```
///
/// Anders als [`LinearDecay`] fällt die Kurve am Anfang und am Ende flach und in der Mitte steil.
/// Das Netz verweilt also lange bei großen und lange bei kleinen Lernraten. Braucht `cos`,
/// deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone, Copy)]
pub struct CosineDecay<T: Scalar = f32> {
    final_fraction: T,
    epochs: usize,
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T: Scalar> CosineDecay<T> {
    /// # Panics
    /// Wenn `epochs` 0 ist.
    pub const fn new(final_fraction: T, epochs: usize) -> Self {
        assert!(epochs > 0, "epochs muss größer als 0 sein");
        Self {
            final_fraction,
            epochs,
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T: Scalar> Schedule<T> for CosineDecay<T> {
    fn rate(&self, epoch: usize, base: T) -> T {
        let progress = T::from_usize(epoch.min(self.epochs)) / T::from_usize(self.epochs);
        let half = T::from_f32(0.5);
        let cosine = half * (T::ONE + (T::from_f64(core::f64::consts::PI) * progress).cos());
        base * (self.final_fraction + (T::ONE - self.final_fraction) * cosine)
    }
}

/// Fährt die Lernrate in den ersten `epochs` Runden langsam hoch und übergibt dann an den
/// Plan `inner`.
///
/// Am Anfang des Trainings sind Gewichte und Statistik (zum Beispiel bei Adam) noch unzuverlässig,
/// große Schritte können das Netz sofort aus der Bahn werfen. Der Anlauf steigert die Lernrate
/// gleichmäßig von `base/epochs` auf `base`. Danach rechnet `inner`, und zwar so, als finge das
/// Training erst jetzt an (Runde 0 von `inner` ist die erste Runde nach dem Anlauf).
///
/// ```
/// use neuronws_layers::schedule::{Constant, Schedule, Warmup};
///
/// let warmup = Warmup::new(4, Constant);
/// let rates: Vec<f32> = (0..6).map(|epoch| warmup.rate(epoch, 0.1)).collect();
/// assert_eq!(rates, [0.025, 0.05, 0.075, 0.1, 0.1, 0.1]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Warmup<S> {
    epochs: usize,
    inner: S,
}

impl<S> Warmup<S> {
    pub const fn new(epochs: usize, inner: S) -> Self {
        Self { epochs, inner }
    }
}

impl<T: Scalar, S: Schedule<T>> Schedule<T> for Warmup<S> {
    fn rate(&self, epoch: usize, base: T) -> T {
        if epoch < self.epochs {
            base * T::from_usize(epoch + 1) / T::from_usize(self.epochs)
        } else {
            self.inner.rate(epoch - self.epochs, base)
        }
    }
}

/// Ein Plan, der nach jeder Runde den **Verlust** sieht und danach die nächste Lernrate festlegt.
///
/// Anders als [`Schedule`] hängt das Ergebnis nicht nur von der Rundennummer ab, sondern vom
/// Verlauf des Trainings. Der Plan hat deshalb ein Gedächtnis und braucht `&mut self`. Benutzt
/// wird er mit `Trainer::fit_adaptive`.
pub trait AdaptiveSchedule<T: Scalar> {
    /// Meldet den Verlust der gerade beendeten Runde und liefert die Lernrate für die nächste.
    /// `current` ist die Lernrate, mit der die Runde gelaufen ist.
    fn next_rate(&mut self, loss: T, current: T) -> T;
}

/// Verkleinert die Lernrate, wenn der Verlust nicht mehr besser wird ("Plateau").
///
/// Der Plan merkt sich den besten Verlust. Wird er `patience` Runden hintereinander nicht um
/// mindestens `min_delta` unterboten, wird die Lernrate mit `factor` multipliziert (aber nicht
/// unter `min_rate`) und das Zählen beginnt von vorn.
///
/// ```
/// use neuronws_layers::schedule::{AdaptiveSchedule, ReduceOnPlateau};
///
/// let mut plateau = ReduceOnPlateau::new(0.5, 2);
/// let mut rate = 0.1_f32;
/// for loss in [1.0, 0.8, 0.8, 0.8, 0.8] {
///     rate = plateau.next_rate(loss, rate);
/// }
/// // Nach 0.8 wurde es zweimal nicht besser (Geduld 2), erst der dritte Versuch zählt als Plateau.
/// assert_eq!(rate, 0.05);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ReduceOnPlateau<T: Scalar = f32> {
    factor: T,
    patience: usize,
    min_delta: T,
    min_rate: T,
    best: Option<T>,
    rounds_without_progress: usize,
}

impl<T: Scalar> ReduceOnPlateau<T> {
    /// `factor` zwischen 0 und 1 (zum Beispiel 0,5 zum Halbieren), `patience` Runden Geduld.
    pub const fn new(factor: T, patience: usize) -> Self {
        Self {
            factor,
            patience,
            min_delta: T::ZERO,
            min_rate: T::ZERO,
            best: None,
            rounds_without_progress: 0,
        }
    }

    /// Eine Verbesserung zählt erst, wenn sie mindestens so groß ist (Standard 0).
    #[must_use]
    pub const fn with_min_delta(mut self, min_delta: T) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// Die Lernrate sinkt nie unter diesen Wert (Standard 0).
    #[must_use]
    pub const fn with_min_rate(mut self, min_rate: T) -> Self {
        self.min_rate = min_rate;
        self
    }

    /// Der beste bisher gemeldete Verlust.
    pub const fn best(&self) -> Option<T> {
        self.best
    }

    /// Wie viele Runden in Folge es keine Verbesserung gab.
    pub const fn rounds_without_progress(&self) -> usize {
        self.rounds_without_progress
    }
}

impl<T: Scalar> AdaptiveSchedule<T> for ReduceOnPlateau<T> {
    fn next_rate(&mut self, loss: T, current: T) -> T {
        let improved = self.best.is_none_or(|best| loss < best - self.min_delta);
        if improved {
            self.best = Some(loss);
            self.rounds_without_progress = 0;
            return current;
        }

        self.rounds_without_progress += 1;
        if self.rounds_without_progress > self.patience {
            self.rounds_without_progress = 0;
            (current * self.factor).max(self.min_rate)
        } else {
            current
        }
    }
}
