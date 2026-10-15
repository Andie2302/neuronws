//! Schichten mit Zufall: [`Dropout`], [`GaussianNoise`] und [`Reparameterize`].
//!
//! Sie zeigen, wie der Zufall in `forward` kommt: Eine [`Operation`] überschreibt
//! [`forward_train`](Operation::forward_train), zieht dort ihre Zufallszahlen und merkt sich in
//! sich selbst, was `backward` braucht (die Maske, das Rauschen). Der Einsatz (`forward`) ist
//! dagegen immer deterministisch, und `Block::forward_mut` (Lernen) ruft `forward_train` auf.
//! Jede dieser Schichten hat ihren **eigenen** Zufallsgenerator, der sich über den Startwert
//! (`seed`) reproduzieren lässt.

use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::{self, Values};

/// Dropout: Beim Lernen wird jeder Wert mit der Wahrscheinlichkeit `rate` auf 0 gesetzt, die
/// übrigen werden mit `1/(1 − rate)` vergrößert. Beim Einsatz passiert nichts.
///
/// Das Netz kann sich dadurch nicht auf einzelne Neuronen verlassen, weil jedes jederzeit
/// ausfallen kann. Es lernt robustere, breiter verteilte Merkmale und überanpasst weniger.
/// Die Vergrößerung ("inverted dropout") sorgt dafür, dass die Werte im Mittel gleich groß
/// bleiben, sodass beim Einsatz nichts umgerechnet werden muss.
///
/// Die Maske wird bei jedem `forward_mut` neu gezogen und für das folgende `backward_mut`
/// gemerkt: Der Gradient fließt nur durch die behaltenen Werte.
///
/// ```
/// use neuronws_layers::dropout::DropoutLayer;
/// use neuronws_layers::traits::{ForwardPath, ForwardPathMut};
///
/// let mut dropout = DropoutLayer::<[f32; 6]>::new(0.5, 42);
/// let input = [1.0; 6];
///
/// // Beim Einsatz unverändert ...
/// assert_eq!(dropout.forward(&input), input);
/// // ... beim Lernen sind manche Werte 0, die anderen 2.
/// let trained = *dropout.forward_mut(&input);
/// assert!(trained.iter().all(|&v| v == 0.0 || v == 2.0));
/// ```
#[derive(Debug, Clone)]
pub struct Dropout<V: Values<T>, T: Scalar = f32> {
    rate: f32,
    rng: Rng,
    /// Die Maske des letzten Lern-Durchgangs: 0 für gestrichene, `1/(1 − rate)` für behaltene Werte.
    mask: V,
    scalar: PhantomData<T>,
}

impl<V: Values<T>, T: Scalar> Dropout<V, T> {
    /// # Panics
    /// Wenn `rate` nicht in `[0, 1)` liegt.
    pub fn new(rate: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&rate),
            "Dropout: rate muss in [0, 1) liegen, nicht {rate}"
        );
        Self {
            rate,
            rng: Rng::new(seed),
            mask: V::zeros(),
            scalar: PhantomData,
        }
    }

    pub const fn rate(&self) -> f32 {
        self.rate
    }

    /// Die Maske des letzten Lern-Durchgangs.
    pub const fn mask(&self) -> &V {
        &self.mask
    }

    /// Startet den Zufall neu, damit ein Lauf sich wiederholen lässt.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }
}

impl<V: Values<T>, T: Scalar> Operation for Dropout<V, T> {
    type Scalar = T;
    type Input = V;
    type Output = V;
    type Params = ();

    /// Einsatz: Dropout tut nichts.
    fn forward(&self, _params: &(), input: &V) -> V {
        *input
    }

    /// Lernen: zieht eine neue Maske und wendet sie an.
    fn forward_train(&mut self, _params: &(), input: &V) -> V {
        let keep = 1.0 - self.rate;
        // Der Faktor wird in `T` gerechnet, damit `f64`-Netze nicht auf `f32`-Genauigkeit fallen.
        let scale = T::ONE / (T::ONE - T::from_f32(self.rate));
        for i in 0..V::LEN {
            *self.mask.value_mut(i) = if self.rng.next_f32() < keep {
                scale
            } else {
                T::ZERO
            };
        }
        values::zip_map(input, &self.mask, |x, m| x * m)
    }

    /// Der Gradient geht nur durch die behaltenen Werte und wird wie sie vergrößert.
    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &V,
        _output: &V,
        grad_output: &V,
    ) -> V {
        values::zip_map(grad_output, &self.mask, |g, m| g * m)
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Eine Dropout-Schicht über die Form `V`.
pub type DropoutLayer<V, T = f32> = Block<Dropout<V, T>>;

impl<V: Values<T>, T: Scalar> Block<Dropout<V, T>> {
    /// # Panics
    /// Wenn `rate` nicht in `[0, 1)` liegt.
    pub fn new(rate: f32, seed: u64) -> Self {
        Self::from_parts(Dropout::new(rate, seed), ())
    }
}

/// Gaußsches Rauschen: Beim Lernen kommt zu jedem Wert eine normalverteilte Zufallszahl mit
/// Streuung `std_dev` dazu. Beim Einsatz passiert nichts.
///
/// Wie Dropout eine Regularisierung: Das Netz sieht jedes Beispiel jedes Mal ein wenig anders und
/// kann es sich nicht auswendig merken. Der Gradient fließt unverändert durch, weil die Ableitung
/// von `x + Rauschen` nach `x` genau 1 ist. Braucht `ln`, `sqrt`, `cos` und `sin` für die
/// normalverteilten Zahlen, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone)]
pub struct GaussianNoise<V: Values<T>, T: Scalar = f32> {
    std_dev: f32,
    rng: Rng,
    shape: PhantomData<(V, T)>,
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<V: Values<T>, T: Scalar> GaussianNoise<V, T> {
    /// # Panics
    /// Wenn `std_dev` negativ ist.
    pub fn new(std_dev: f32, seed: u64) -> Self {
        assert!(
            std_dev >= 0.0,
            "GaussianNoise: std_dev darf nicht negativ sein"
        );
        Self {
            std_dev,
            rng: Rng::new(seed),
            shape: PhantomData,
        }
    }

    /// Startet den Zufall neu, damit ein Lauf sich wiederholen lässt.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<V: Values<T>, T: Scalar> Operation for GaussianNoise<V, T> {
    type Scalar = T;
    type Input = V;
    type Output = V;
    type Params = ();

    fn forward(&self, _params: &(), input: &V) -> V {
        *input
    }

    fn forward_train(&mut self, _params: &(), input: &V) -> V {
        let std_dev = T::from_f32(self.std_dev);
        values::map(input, |x| x + std_dev * T::from_f32(self.rng.next_normal()))
    }

    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &V,
        _output: &V,
        grad_output: &V,
    ) -> V {
        *grad_output
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Eine Rausch-Schicht über die Form `V`.
#[cfg(any(feature = "std", feature = "libm"))]
pub type GaussianNoiseLayer<V, T = f32> = Block<GaussianNoise<V, T>>;

#[cfg(any(feature = "std", feature = "libm"))]
impl<V: Values<T>, T: Scalar> Block<GaussianNoise<V, T>> {
    /// # Panics
    /// Wenn `std_dev` negativ ist.
    pub fn new(std_dev: f32, seed: u64) -> Self {
        Self::from_parts(GaussianNoise::new(std_dev, seed), ())
    }
}

/// Der Reparametrisierungs-Trick eines Variational Autoencoders (VAE): zieht einen Wert `z` aus
/// der Normalverteilung mit Mittelwert `μ` und Logarithmus der Varianz `log σ²`, **ohne** dass der
/// Zufall den Gradienten blockiert.
///
/// ```text
/// z = μ + σ · ε      mit  σ = e^(½ log σ²)  und  ε ~ N(0, 1)
/// ```
///
/// Die Eingabe ist das Paar `(μ, log σ²)`, wie es ein Encoder mit zwei Ausgängen liefert (zum
/// Beispiel über `Fork` aus zwei Dense-Schichten). Der Zufall steckt allein in `ε`, das beim
/// Lernen gezogen und gemerkt wird. Alles andere ist glatt ableitbar:
/// `∂z/∂μ = 1` und `∂z/∂(log σ²) = ½ σ ε`. Beim Einsatz gibt es kein `ε`, `z = μ`.
///
/// Braucht `exp` und normalverteilte Zahlen, deshalb nur mit dem Feature `std` oder `libm`.
#[cfg(any(feature = "std", feature = "libm"))]
#[derive(Debug, Clone)]
pub struct Reparameterize<V: Values<T>, T: Scalar = f32> {
    rng: Rng,
    /// Das `ε` des letzten Lern-Durchgangs.
    noise: V,
    scalar: PhantomData<T>,
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<V: Values<T>, T: Scalar> Reparameterize<V, T> {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            noise: V::zeros(),
            scalar: PhantomData,
        }
    }

    /// Das `ε` des letzten Lern-Durchgangs.
    pub const fn noise(&self) -> &V {
        &self.noise
    }

    /// Startet den Zufall neu, damit ein Lauf sich wiederholen lässt.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    /// `σ = e^(½ log σ²)`
    fn deviation(log_variance: T) -> T {
        (T::from_f32(0.5) * log_variance).exp()
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<V: Values<T>, T: Scalar> Operation for Reparameterize<V, T> {
    type Scalar = T;
    type Input = (V, V);
    type Output = V;
    type Params = ();

    /// Einsatz: der Mittelwert, ohne Zufall.
    fn forward(&self, _params: &(), (mean, _): &(V, V)) -> V {
        *mean
    }

    fn forward_train(&mut self, _params: &(), (mean, log_variance): &(V, V)) -> V {
        for i in 0..V::LEN {
            *self.noise.value_mut(i) = T::from_f32(self.rng.next_normal());
        }
        values::from_fn(|i| {
            mean.value(i) + Self::deviation(log_variance.value(i)) * self.noise.value(i)
        })
    }

    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        (_, log_variance): &(V, V),
        _output: &V,
        grad_output: &V,
    ) -> (V, V) {
        let half = T::from_f32(0.5);
        (
            *grad_output,
            values::from_fn(|i| {
                grad_output.value(i)
                    * half
                    * Self::deviation(log_variance.value(i))
                    * self.noise.value(i)
            }),
        )
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Eine Reparametrisierungs-Schicht über die Form `V`.
#[cfg(any(feature = "std", feature = "libm"))]
pub type ReparameterizeLayer<V, T = f32> = Block<Reparameterize<V, T>>;

#[cfg(any(feature = "std", feature = "libm"))]
impl<V: Values<T>, T: Scalar> Block<Reparameterize<V, T>> {
    pub fn new(seed: u64) -> Self {
        Self::from_parts(Reparameterize::new(seed), ())
    }
}
