/// Eine Verlustfunktion: misst, wie weit die Ausgabe vom Ziel entfernt ist.
///
/// `gradient` ist `∂L/∂y` und damit der Startpunkt des Backward-Passes.
pub trait Loss<const N: usize> {
    fn value(&self, output: &[f32; N], target: &[f32; N]) -> f32;
    fn gradient(&self, output: &[f32; N], target: &[f32; N]) -> [f32; N];
}

/// Mittlerer quadratischer Fehler: `L = (1/N) Σ (y − t)²`.
pub struct MeanSquaredError;

impl<const N: usize> Loss<N> for MeanSquaredError {
    fn value(&self, output: &[f32; N], target: &[f32; N]) -> f32 {
        let sum: f32 = output
            .iter()
            .zip(target)
            .map(|(y, t)| (y - t) * (y - t))
            .sum();
        sum / N as f32
    }

    /// `∂L/∂y[i] = 2 (y[i] − t[i]) / N`
    fn gradient(&self, output: &[f32; N], target: &[f32; N]) -> [f32; N] {
        core::array::from_fn(|i| 2.0 * (output[i] - target[i]) / N as f32)
    }
}
