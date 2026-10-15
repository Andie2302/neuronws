use crate::scalar::Scalar;
use crate::traits::{
    BackwardData, BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer, Trainable,
};

/// Standardisiert jede Eingabe einzeln: `x' = (x − Mittelwert) / Streuung`.
///
/// Danach hat jede Eingabe im Mittel den Wert 0 und die Streuung 1, egal in welchen Einheiten die
/// Rohdaten vorlagen (Pixel von 0 bis 1 oder von 0 bis 100, Grad Celsius, Millimeter, ...).
/// Das ist die Voraussetzung dafür, dass Aktivierungen und Lernrate zur Größe der Daten passen.
///
/// Die Statistik wird **einmal** aus den Lerndaten berechnet (`fit`) und dann festgehalten. Sie ist
/// kein lernbarer Parameter, sondern Teil des Modells: Beim Anwenden müssen neue Daten genauso
/// skaliert werden wie die Lerndaten, und weil die Schicht vor dem Netz sitzt, passiert das von
/// allein.
///
/// Eine Eingabe, die in den Lerndaten konstant ist (Streuung 0), wird nur um den Mittelwert
/// verschoben und nicht skaliert. So gibt es nie eine Division durch 0.
///
/// ```
/// use neuronws_layers::standardize::Standardize;
/// use neuronws_layers::traits::ForwardPath;
///
/// // Zwei Eingaben in sehr verschiedenen Größenordnungen (z. B. Meter und Millimeter)
/// let data = [[1.0_f32, 1000.0], [2.0, 2000.0], [3.0, 3000.0]];
/// # #[cfg(any(feature = "std", feature = "libm"))]
/// # {
/// let standardize = Standardize::fit(&data);
/// let [a, b] = standardize.forward(&[2.0, 2000.0]);
/// assert!(a.abs() < 1e-6 && b.abs() < 1e-6); // der Mittelwert wird zu 0
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Standardize<const N: usize, T: Scalar = f32> {
    mean: [T; N],
    std_dev: [T; N],
    output: [T; N],
    grad_input: [T; N],
}

impl<const N: usize, T: Scalar> Standardize<N, T> {
    /// Aus bekannter Statistik. Eine Streuung `≤ 0` (oder keine Zahl) wird wie 1 behandelt.
    pub fn from_statistics(mean: [T; N], std_dev: [T; N]) -> Self {
        Self {
            mean,
            std_dev: std_dev.map(|s| if s > T::ZERO { s } else { T::ONE }),
            output: [T::ZERO; N],
            grad_input: [T::ZERO; N],
        }
    }

    /// Berechnet Mittelwert und Streuung (der Lerndaten, geteilt durch ihre Anzahl) in einem Durchgang.
    ///
    /// Braucht die Wurzel, deshalb nur mit dem Feature `std` oder `libm`.
    ///
    /// # Panics
    /// Wenn `samples` leer ist.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn fit<'a>(samples: impl IntoIterator<Item = &'a [T; N]>) -> Self
    where
        T: 'a,
    {
        // Welford-Verfahren: numerisch stabil und ohne die Daten zweimal zu lesen.
        let mut count = 0_usize;
        let mut mean = [T::ZERO; N];
        let mut sum_squared_deviations = [T::ZERO; N];
        for sample in samples {
            count += 1;
            let n = T::from_usize(count);
            for i in 0..N {
                let delta = sample[i] - mean[i];
                mean[i] += delta / n;
                sum_squared_deviations[i] += delta * (sample[i] - mean[i]);
            }
        }
        assert!(
            count > 0,
            "Standardize::fit braucht mindestens ein Beispiel"
        );

        let n = T::from_usize(count);
        Self::from_statistics(mean, sum_squared_deviations.map(|s| (s / n).sqrt()))
    }

    pub const fn mean(&self) -> &[T; N] {
        &self.mean
    }

    /// Die Streuung, mit der geteilt wird (1 für Eingaben, die in den Lerndaten konstant waren).
    pub const fn std_dev(&self) -> &[T; N] {
        &self.std_dev
    }
}

impl<const N: usize, T: Scalar> Layer for Standardize<N, T> {}

impl<const N: usize, T: Scalar> ForwardData for Standardize<N, T> {
    type ForwardInput = [T; N];
    type ForwardOutput = [T; N];
}

impl<const N: usize, T: Scalar> ForwardPath for Standardize<N, T> {
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        core::array::from_fn(|i| (input[i] - self.mean[i]) / self.std_dev[i])
    }
}

impl<const N: usize, T: Scalar> ForwardPathMut for Standardize<N, T> {
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        self.output = self.forward(input);
        &self.output
    }
}

impl<const N: usize, T: Scalar> BackwardData for Standardize<N, T> {
    type BackwardInput = [T; N];
    type BackwardOutput = [T; N];
}

impl<const N: usize, T: Scalar> BackwardPathMut for Standardize<N, T> {
    /// `∂L/∂x[i] = g[i] / Streuung[i]`
    fn backward_mut(&mut self, grad_output: &Self::BackwardInput) -> &Self::BackwardOutput {
        for ((grad_x, &g), &s) in self
            .grad_input
            .iter_mut()
            .zip(grad_output)
            .zip(&self.std_dev)
        {
            *grad_x = g / s;
        }
        &self.grad_input
    }
}

/// Mittelwert und Streuung sind fest, es gibt nichts zu lernen.
impl<const N: usize, T: Scalar> Trainable for Standardize<N, T> {
    type Scalar = T;

    const PARAMETER_COUNT: usize = 0;

    fn visit_parameters(&mut self, _visitor: &mut impl FnMut(&mut T, T)) {}

    fn zero_grad(&mut self) {}
}
