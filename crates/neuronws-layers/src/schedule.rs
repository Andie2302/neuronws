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
