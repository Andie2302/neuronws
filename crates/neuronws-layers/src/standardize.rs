use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::Values;

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
/// Die Rechnung dazu: Mittelwert und Streuung sind feste Einstellungen der Operation, keine
/// lernbaren Parameter.
#[derive(Debug, Clone, Copy)]
pub struct Standardization<const N: usize, T: Scalar = f32> {
    mean: [T; N],
    std_dev: [T; N],
}

impl<const N: usize, T: Scalar> Operation for Standardization<N, T> {
    type Scalar = T;
    type Input = [T; N];
    type Output = [T; N];
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        core::array::from_fn(|i| (input[i] - self.mean[i]) / self.std_dev[i])
    }

    /// `∂L/∂x[i] = g[i] / Streuung[i]`
    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        core::array::from_fn(|i| grad_output[i] / self.std_dev[i])
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Standardisiert jede Eingabe einzeln, siehe [`Standardization`].
pub type Standardize<const N: usize, T = f32> = Block<Standardization<N, T>>;

impl<const N: usize, T: Scalar> Block<Standardization<N, T>> {
    /// Aus bekannter Statistik. Eine Streuung `≤ 0` (oder keine Zahl) wird wie 1 behandelt.
    pub fn from_statistics(mean: [T; N], std_dev: [T; N]) -> Self {
        let std_dev = std_dev.map(|s| if s > T::ZERO { s } else { T::ONE });
        Self::from_parts(Standardization { mean, std_dev }, ())
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
        &self.operation().mean
    }

    /// Die Streuung, mit der geteilt wird (1 für Eingaben, die in den Lerndaten konstant waren).
    pub const fn std_dev(&self) -> &[T; N] {
        &self.operation().std_dev
    }
}

/// Standardisiert **pro Kanal**: Alle Werte eines Kanals teilen sich einen Mittelwert und eine
/// Streuung.
///
/// Die äußerste Achse von `[V; N]` zählt als Kanalachse, `V` ist die Form eines Kanals. Bei einem
/// Bild `[[[T; W]; H]; C]` hat also jeder der `C` Kanäle seine eigene Statistik, gebildet über
/// alle Pixel des Kanals und alle Lernbeispiele (so wie man Farbbilder normalisiert: ein
/// Mittelwert für Rot, einer für Grün, einer für Blau). [`Standardization`] ist der Sonderfall, in
/// dem jeder Kanal nur aus einer Zahl besteht.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))]
/// # {
/// use neuronws_layers::standardize::StandardizeChannels;
/// use neuronws_layers::traits::ForwardPath;
///
/// // Zwei Kanäle von 1×2 Pixeln: der erste um 10 herum, der zweite um 1000.
/// let images = [
///     [[[9.0_f32, 11.0]], [[990.0, 1010.0]]],
///     [[[10.0, 10.0]], [[1000.0, 1000.0]]],
/// ];
/// let standardize = StandardizeChannels::<2, 1, 2>::fit(&images);
/// let [[[a, _]], [[b, _]]] = standardize.forward(&images[1]);
/// assert!(a.abs() < 1e-5 && b.abs() < 1e-5); // beide Kanäle: Mittelwert wird 0
/// # }
/// ```
///
/// Wie bei [`Standardization`] sind Mittelwert und Streuung feste Einstellungen und keine lernbaren
/// Parameter. Ein Kanal ohne Streuung wird nur verschoben, nicht skaliert.
#[derive(Debug, Clone, Copy)]
pub struct ChannelStandardization<V: Values<T>, const N: usize, T: Scalar = f32> {
    mean: [T; N],
    std_dev: [T; N],
    shape: PhantomData<V>,
}

impl<V: Values<T>, const N: usize, T: Scalar> Operation for ChannelStandardization<V, N, T> {
    type Scalar = T;
    type Input = [V; N];
    type Output = [V; N];
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        core::array::from_fn(|c| {
            crate::values::map(&input[c], |x| (x - self.mean[c]) / self.std_dev[c])
        })
    }

    /// `∂L/∂x = g / Streuung[kanal]`
    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        core::array::from_fn(|c| crate::values::map(&grad_output[c], |g| g / self.std_dev[c]))
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Standardisiert die `C` Kanäle eines Bildes einzeln, siehe [`ChannelStandardization`].
pub type StandardizeChannels<const C: usize, const H: usize, const W: usize, T = f32> =
    Block<ChannelStandardization<[[T; W]; H], C, T>>;

impl<V: Values<T>, const N: usize, T: Scalar> Block<ChannelStandardization<V, N, T>> {
    /// Aus bekannter Statistik. Eine Streuung `≤ 0` (oder keine Zahl) wird wie 1 behandelt.
    pub fn from_statistics(mean: [T; N], std_dev: [T; N]) -> Self {
        let std_dev = std_dev.map(|s| if s > T::ZERO { s } else { T::ONE });
        Self::from_parts(
            ChannelStandardization {
                mean,
                std_dev,
                shape: PhantomData,
            },
            (),
        )
    }

    /// Berechnet Mittelwert und Streuung jedes Kanals über alle Beispiele und alle Stellen des
    /// Kanals in einem Durchgang.
    ///
    /// Braucht die Wurzel, deshalb nur mit dem Feature `std` oder `libm`.
    ///
    /// # Panics
    /// Wenn `samples` leer ist.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn fit<'a>(samples: impl IntoIterator<Item = &'a [V; N]>) -> Self
    where
        V: 'a,
        T: 'a,
    {
        // Welford-Verfahren, jeder Kanal sammelt die Werte aller seiner Stellen.
        let mut count = 0_usize;
        let mut mean = [T::ZERO; N];
        let mut sum_squared_deviations = [T::ZERO; N];
        let mut any_sample = false;
        for sample in samples {
            any_sample = true;
            for position in 0..V::LEN {
                count += 1;
                let n = T::from_usize(count);
                // `count` zählt für alle Kanäle gleich weiter: pro Stelle und Beispiel ein Wert
                // je Kanal. Deshalb dürfen alle Kanäle in derselben Runde denselben Zähler
                // benutzen.
                for c in 0..N {
                    let x = sample[c].value(position);
                    let delta = x - mean[c];
                    mean[c] += delta / n;
                    sum_squared_deviations[c] += delta * (x - mean[c]);
                }
            }
        }
        assert!(
            any_sample,
            "StandardizeChannels::fit braucht mindestens ein Beispiel"
        );

        let n = T::from_usize(count);
        Self::from_statistics(mean, sum_squared_deviations.map(|s| (s / n).sqrt()))
    }

    pub const fn mean(&self) -> &[T; N] {
        &self.operation().mean
    }

    /// Die Streuung, mit der geteilt wird (1 für Kanäle, die in den Lerndaten konstant waren).
    pub const fn std_dev(&self) -> &[T; N] {
        &self.operation().std_dev
    }
}
