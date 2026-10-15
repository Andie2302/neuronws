//! Normalisierungsschichten: [`LayerNorm`] und [`BatchNorm`].
//!
//! Beide bringen die Werte einer Schicht auf Mittelwert 0 und Streuung 1 und geben dem Netz mit
//! zwei lernbaren Parametern `γ` (Skalierung) und `β` (Verschiebung) die Freiheit zurück, davon
//! wieder abzuweichen: `y = γ · x̂ + β`. Das hält die Größenordnung der Werte von Schicht zu
//! Schicht stabil und macht das Lernen unempfindlicher gegen Startwerte und Lernrate.
//!
//! Der Unterschied liegt darin, **woher** Mittelwert und Streuung kommen:
//!
//! | | Statistik aus | Beim Einsatz |
//! |---|---|---|
//! | [`LayerNorm`] | den Werten **dieses einen** Beispiels | derselbe Weg |
//! | [`BatchNorm`] | einem laufenden Durchschnitt über viele Beispiele | feste Statistik |
//!
//! Beide brauchen die Wurzel, deshalb gibt es sie nur mit dem Feature `std` oder `libm`.

use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::{self, Values};

/// Layer-Normalisierung: Jedes Beispiel wird für sich über **alle** seine Werte normalisiert.
///
/// ```text
/// μ = Mittelwert(x)     σ² = Varianz(x)     x̂ = (x − μ) / √(σ² + ε)     y = γ · x̂ + β
/// ```
///
/// `V` ist die Form, über die normalisiert wird (ein Vektor, eine Matrix, ein ganzes Bild), und
/// `γ` und `β` haben dieselbe Form: ein eigener Faktor und eine eigene Verschiebung pro Wert.
/// Weil die Statistik nur vom einzelnen Beispiel abhängt, rechnet die Schicht beim Lernen und
/// beim Einsatz gleich, und die Batchgröße spielt keine Rolle. Das ist der Vorteil gegenüber
/// [`BatchNorm`], und der Grund, warum Layer-Normalisierung bei Folgen (RNN, Transformer) üblich
/// ist.
///
/// Am Anfang ist `γ = 1` und `β = 0`: Die Schicht normalisiert nur. `ε` (Standard 1e-5) verhindert
/// die Division durch 0, wenn alle Werte gleich sind.
///
/// ```
/// use neuronws_layers::norm::LayerNormLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// let layer = LayerNormLayer::<[f64; 4], f64>::new();
/// let y = layer.forward(&[1.0, 2.0, 3.0, 4.0]);
/// let mean: f64 = y.iter().sum::<f64>() / 4.0;
/// let variance: f64 = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
/// assert!(mean.abs() < 1e-9);
/// assert!((variance - 1.0).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LayerNorm<V: Values<T>, T: Scalar = f32> {
    epsilon: T,
    shape: PhantomData<V>,
}

impl<V: Values<T>, T: Scalar> LayerNorm<V, T> {
    pub fn new() -> Self {
        Self {
            epsilon: T::from_f32(1e-5),
            shape: PhantomData,
        }
    }

    /// Ein eigenes `ε`.
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Mittelwert und `1/√(σ² + ε)` der Werte von `input`.
    fn statistics(&self, input: &V) -> (T, T) {
        let n = T::from_usize(V::LEN);
        let mean = values::sum(input) / n;
        let variance = (0..V::LEN).fold(T::ZERO, |sum, i| {
            let d = input.value(i) - mean;
            sum + d * d
        }) / n;
        (mean, T::ONE / (variance + self.epsilon).sqrt())
    }
}

impl<V: Values<T>, T: Scalar> Default for LayerNorm<V, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Values<T>, T: Scalar> Operation for LayerNorm<V, T> {
    type Scalar = T;
    type Input = V;
    type Output = V;
    /// `(γ, β)`
    type Params = (V, V);

    fn forward(&self, (gamma, beta): &Self::Params, input: &V) -> V {
        let (mean, inv_std) = self.statistics(input);
        values::from_fn(|i| gamma.value(i) * (input.value(i) - mean) * inv_std + beta.value(i))
    }

    /// Mit `x̂ = (x − μ)/σ` und `dx̂ = g · γ`:
    /// `∂L/∂x = (dx̂ − Mittelwert(dx̂) − x̂ · Mittelwert(dx̂ · x̂)) / σ`
    ///
    /// Die beiden Mittelwerte entstehen, weil `μ` und `σ` selbst von jedem `x` abhängen.
    fn backward(
        &self,
        (gamma, _): &Self::Params,
        (grad_gamma, grad_beta): &mut Self::Params,
        input: &V,
        _output: &V,
        grad_output: &V,
    ) -> V {
        let (mean, inv_std) = self.statistics(input);
        let n = T::from_usize(V::LEN);
        let normalized: V = values::from_fn(|i| (input.value(i) - mean) * inv_std);
        let grad_normalized: V = values::from_fn(|i| grad_output.value(i) * gamma.value(i));

        for i in 0..V::LEN {
            *grad_gamma.value_mut(i) += grad_output.value(i) * normalized.value(i);
            *grad_beta.value_mut(i) += grad_output.value(i);
        }

        let mean_grad = values::sum(&grad_normalized) / n;
        let mean_grad_times_normalized =
            values::sum(&values::zip_map(&grad_normalized, &normalized, |a, b| {
                a * b
            })) / n;
        values::from_fn(|i| {
            inv_std
                * (grad_normalized.value(i)
                    - mean_grad
                    - normalized.value(i) * mean_grad_times_normalized)
        })
    }

    /// `γ = 1`, `β = 0`, egal welcher `Initializer` übergeben wird.
    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) -> Self::Params {
        (values::from_fn(|_| T::ONE), V::zeros())
    }
}

/// Eine Layer-Normalisierung über die Form `V`.
pub type LayerNormLayer<V, T = f32> = Block<LayerNorm<V, T>>;

impl<V: Values<T>, T: Scalar> Block<LayerNorm<V, T>> {
    pub fn new() -> Self {
        Self::from_parts(LayerNorm::new(), (values::from_fn(|_| T::ONE), V::zeros()))
    }

    /// Aus fertigen `γ` und `β`.
    pub fn with_parameters(gamma: V, beta: V) -> Self {
        Self::from_parts(LayerNorm::new(), (gamma, beta))
    }

    pub const fn gamma(&self) -> &V {
        &self.params().0
    }

    pub const fn beta(&self) -> &V {
        &self.params().1
    }
}

impl<V: Values<T>, T: Scalar> Default for Block<LayerNorm<V, T>> {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch-Normalisierung mit **laufender Statistik**: Mittelwert und Streuung pro Kanal kommen aus
/// einem gleitenden Durchschnitt über viele Beispiele.
///
/// ```text
/// x̂ = (x − Mittelwert_laufend) / √(Varianz_laufend + ε)         y = γ · x̂ + β
/// ```
///
/// Die äußerste Achse von `[V; N]` ist die Kanalachse: Jeder der `N` Kanäle hat seine eigene
/// Statistik und sein eigenes `γ` und `β`, gebildet über alle Stellen von `V` und alle Beispiele.
/// Für Vektoren steht jeder Eintrag für einen Kanal ([`BatchNorm1d`]), für Bilder jeder
/// Farbkanal ([`BatchNorm2d`]).
///
/// # Was anders ist als bei der klassischen Batch-Normalisierung
///
/// Die klassische Variante normalisiert beim Lernen mit Mittelwert und Streuung des **aktuellen
/// Batches** und führt daneben einen gleitenden Durchschnitt für den Einsatz mit. Diese Bibliothek
/// verarbeitet die Beispiele eines Batches aber **eins nach dem anderen** und sammelt nur die
/// Gradienten. Während das erste Beispiel durchläuft, sind die anderen noch gar nicht da. Deshalb
/// gilt hier schon beim Lernen der gleitende Durchschnitt, der mit jedem Beispiel
/// (`forward_mut`) nachgeführt wird (`momentum`, Standard 0,1). Beim Einsatz (`forward`) bleibt
/// er unverändert. Zwei Folgen:
///
/// - Die Statistik wird als Konstante behandelt, der Gradient fließt nicht durch sie hindurch.
///   Das ist bei langsam wandernder Statistik eine gute Näherung, und die Rückwärtsrechnung ist
///   exakt zu dem, was `forward` rechnet (der Gradienten-Check besteht).
/// - Am Anfang (Mittelwert 0, Varianz 1) normalisiert die Schicht noch kaum. Sie lernt die
///   Statistik in den ersten Runden. Wer die Daten kennt, kann sie mit
///   [`with_statistics`](Block::with_statistics) direkt vorgeben.
///
/// Die laufende Statistik gehört **nicht** zu den lernbaren Parametern. Sie steht also nicht in
/// `parameters()` und wird vom Optimierer nicht angefasst. Wer ein trainiertes Netz sichert,
/// sichert sie mit [`running_mean`](Block::running_mean) und [`running_var`](Block::running_var).
#[derive(Debug, Clone, Copy)]
pub struct BatchNorm<V: Values<T>, const N: usize, T: Scalar = f32> {
    momentum: T,
    epsilon: T,
    running_mean: [T; N],
    running_var: [T; N],
    shape: PhantomData<V>,
}

impl<V: Values<T>, const N: usize, T: Scalar> BatchNorm<V, N, T> {
    fn new() -> Self {
        Self {
            momentum: T::from_f32(0.1),
            epsilon: T::from_f32(1e-5),
            running_mean: [T::ZERO; N],
            running_var: [T::ONE; N],
            shape: PhantomData,
        }
    }

    /// Normalisiert mit der aktuellen laufenden Statistik.
    fn normalize(&self, (gamma, beta): &<Self as Operation>::Params, input: &[V; N]) -> [V; N] {
        core::array::from_fn(|c| {
            let inv_std = T::ONE / (self.running_var[c] + self.epsilon).sqrt();
            values::map(&input[c], |x| {
                gamma[c] * (x - self.running_mean[c]) * inv_std + beta[c]
            })
        })
    }

    /// Führt die laufende Statistik um ein weiteres Beispiel nach.
    ///
    /// Jedes Beispiel ist eine kleine Verteilung (Mittelwert `m`, Varianz `v` über die Stellen
    /// des Kanals). Der gleitende Durchschnitt mischt sie mit Gewicht `momentum` zur alten:
    /// `μ ← μ + α (m − μ)` und `σ² ← (1 − α) (σ² + α (m − μ)²) + α v`.
    fn update_statistics(&mut self, input: &[V; N]) {
        let alpha = self.momentum;
        let len = T::from_usize(V::LEN);
        for (c, channel) in input.iter().enumerate() {
            let sample_mean = values::sum(channel) / len;
            let sample_var = (0..V::LEN).fold(T::ZERO, |sum, i| {
                let d = channel.value(i) - sample_mean;
                sum + d * d
            }) / len;
            let delta = sample_mean - self.running_mean[c];
            self.running_mean[c] += alpha * delta;
            self.running_var[c] = (T::ONE - alpha) * (self.running_var[c] + alpha * delta * delta)
                + alpha * sample_var;
        }
    }
}

impl<V: Values<T>, const N: usize, T: Scalar> Operation for BatchNorm<V, N, T> {
    type Scalar = T;
    type Input = [V; N];
    type Output = [V; N];
    /// `(γ, β)`, je eines pro Kanal.
    type Params = ([T; N], [T; N]);

    /// Einsatz: die laufende Statistik bleibt, wie sie ist.
    fn forward(&self, params: &Self::Params, input: &[V; N]) -> [V; N] {
        self.normalize(params, input)
    }

    /// Lernen: erst die Statistik um dieses Beispiel nachführen, dann damit normalisieren.
    fn forward_train(&mut self, params: &Self::Params, input: &[V; N]) -> [V; N] {
        self.update_statistics(input);
        self.normalize(params, input)
    }

    /// Die Statistik ist hier eine Konstante: `∂L/∂x = g · γ / σ`, `∂L/∂γ = Σ g · x̂`,
    /// `∂L/∂β = Σ g` (summiert über alle Stellen des Kanals).
    fn backward(
        &self,
        (gamma, _): &Self::Params,
        (grad_gamma, grad_beta): &mut Self::Params,
        input: &[V; N],
        _output: &[V; N],
        grad_output: &[V; N],
    ) -> [V; N] {
        core::array::from_fn(|c| {
            let inv_std = T::ONE / (self.running_var[c] + self.epsilon).sqrt();
            for i in 0..V::LEN {
                let normalized = (input[c].value(i) - self.running_mean[c]) * inv_std;
                grad_gamma[c] += grad_output[c].value(i) * normalized;
                grad_beta[c] += grad_output[c].value(i);
            }
            values::map(&grad_output[c], |g| g * gamma[c] * inv_std)
        })
    }

    /// `γ = 1`, `β = 0`, egal welcher `Initializer` übergeben wird.
    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) -> Self::Params {
        ([T::ONE; N], [T::ZERO; N])
    }
}

/// Batch-Normalisierung für Vektoren: jeder der `N` Einträge ist ein Kanal.
pub type BatchNorm1d<const N: usize, T = f32> = BatchNorm<T, N, T>;

/// Batch-Normalisierung für Bilder mit `C` Kanälen zu je `H × W` Pixeln.
pub type BatchNorm2d<const C: usize, const H: usize, const W: usize, T = f32> =
    BatchNorm<[[T; W]; H], C, T>;

/// Die Schicht zu [`BatchNorm1d`].
pub type BatchNorm1dLayer<const N: usize, T = f32> = Block<BatchNorm<T, N, T>>;

/// Die Schicht zu [`BatchNorm2d`].
pub type BatchNorm2dLayer<const C: usize, const H: usize, const W: usize, T = f32> =
    Block<BatchNorm<[[T; W]; H], C, T>>;

impl<V: Values<T>, const N: usize, T: Scalar> Block<BatchNorm<V, N, T>> {
    /// `γ = 1`, `β = 0`, laufender Mittelwert 0 und laufende Varianz 1.
    pub fn new() -> Self {
        Self::from_parts(BatchNorm::new(), ([T::ONE; N], [T::ZERO; N]))
    }

    /// Mit bekannter Statistik als Startwert (zum Beispiel aus den Lerndaten oder aus einem
    /// früheren Training).
    #[must_use]
    pub fn with_statistics(mut self, mean: [T; N], variance: [T; N]) -> Self {
        let op = self.operation_mut();
        op.running_mean = mean;
        op.running_var = variance;
        self
    }

    /// Das Gewicht, mit dem jedes neue Beispiel in die laufende Statistik eingeht (Standard 0,1).
    /// Kleiner heißt träger und ruhiger, größer heißt schneller angepasst.
    #[must_use]
    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.operation_mut().momentum = momentum;
        self
    }

    pub const fn running_mean(&self) -> &[T; N] {
        &self.operation().running_mean
    }

    pub const fn running_var(&self) -> &[T; N] {
        &self.operation().running_var
    }

    pub const fn gamma(&self) -> &[T; N] {
        &self.params().0
    }

    pub const fn beta(&self) -> &[T; N] {
        &self.params().1
    }
}

impl<V: Values<T>, const N: usize, T: Scalar> Default for Block<BatchNorm<V, N, T>> {
    fn default() -> Self {
        Self::new()
    }
}
