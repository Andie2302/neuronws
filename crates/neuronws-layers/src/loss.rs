use crate::scalar::Scalar;

/// Eine Verlustfunktion: misst, wie weit die Ausgabe vom Ziel entfernt ist.
///
/// `gradient` ist `∂L/∂y` und damit der Startpunkt des Backward-Passes.
pub trait Loss<const N: usize> {
    fn value<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> T;
    fn gradient<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> [T; N];
}

/// Mittlerer quadratischer Fehler: `L = (1/N) Σ (y − t)²`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanSquaredError;

impl<const N: usize> Loss<N> for MeanSquaredError {
    fn value<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> T {
        let sum = output
            .iter()
            .zip(target)
            .fold(T::ZERO, |sum, (&y, &t)| sum + (y - t) * (y - t));
        sum / T::from_usize(N)
    }

    /// `∂L/∂y[i] = 2 (y[i] − t[i]) / N`
    fn gradient<T: Scalar>(&self, output: &[T; N], target: &[T; N]) -> [T; N] {
        core::array::from_fn(|i| T::TWO * (output[i] - target[i]) / T::from_usize(N))
    }
}
