use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::{Image, Values};

/// Die Kerne einer Faltung: `kernels[ausgabekanal][eingabekanal][zeile][spalte]`.
pub type Kernels<const C_OUT: usize, const C_IN: usize, const KH: usize, const KW: usize, T = f32> =
    [[[[T; KW]; KH]; C_IN]; C_OUT];

/// Zweidimensionale Faltung ("valid", Schrittweite 1, ohne Rand).
///
/// Ein kleiner Kern der Größe `KH × KW` wird über das Bild geschoben. An jeder Stelle wird die
/// Summe aus Kern mal darunterliegenden Bildwerten (über alle Eingabekanäle) plus Bias gebildet:
///
/// ```text
/// y[o][r][c] = b[o] + Σ_{i,p,q} K[o][i][p][q] · x[i][r + p][c + q]
/// ```
///
/// Dieselben Kernwerte gelten an jeder Stelle. Das spart Parameter und erkennt ein Muster
/// überall im Bild. Beim Zurückrechnen werden die Beiträge aller Stellen zu den Kernwerten
/// aufsummiert.
///
/// Die Ausgabegröße ergibt sich aus der Eingabe: `OH = H − KH + 1` und `OW = W − KW + 1`. Rust
/// kann das noch nicht selbst ausrechnen, deshalb steht sie als eigene Zahl da. Stimmt sie nicht,
/// bricht das Kompilieren ab. Der Fehler erscheint erst beim Bauen (`cargo build`, `cargo test`),
/// `cargo check` und die Anzeige in der IDE bemerken ihn nicht:
///
/// ```compile_fail
/// use neuronws_layers::conv::Conv2d;
///
/// // 5×5-Bild mit 3×3-Kern ergibt 3×3, nicht 4×4.
/// let _ = Conv2d::<1, 1, 5, 5, 3, 3, 4, 4>::new();
/// ```
///
/// ```
/// use neuronws_layers::conv::Conv2dLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// // Ein Kanal rein, ein Kanal raus, 3×3-Bild, 2×2-Kern, 2×2 heraus.
/// let kernel = [[[[1.0, 0.0], [0.0, -1.0]]]];
/// let conv = Conv2dLayer::<1, 1, 3, 3, 2, 2, 2, 2>::new(kernel, [0.0]);
/// let image = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]];
/// assert_eq!(conv.forward(&image), [[[-4.0, -4.0], [-4.0, -4.0]]]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Conv2d<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar = f32,
> {
    scalar: PhantomData<T>,
}

impl<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Conv2d<C_IN, C_OUT, H, W, KH, KW, OH, OW, T>
{
    /// # Compile-Fehler
    /// Wenn der Kern größer als das Bild ist oder `OH`/`OW` nicht `H − KH + 1`/`W − KW + 1` sind.
    pub const fn new() -> Self {
        const {
            assert!(
                KH >= 1 && KW >= 1 && KH <= H && KW <= W,
                "Conv2d: der Kern muss mindestens 1×1 und höchstens so groß wie das Bild sein"
            );
            assert!(
                OH == H + 1 - KH && OW == W + 1 - KW,
                "Conv2d: die Ausgabegröße muss (H − KH + 1) × (W − KW + 1) sein"
            );
        };
        Self {
            scalar: PhantomData,
        }
    }
}

impl<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Default for Conv2d<C_IN, C_OUT, H, W, KH, KW, OH, OW, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Operation for Conv2d<C_IN, C_OUT, H, W, KH, KW, OH, OW, T>
{
    type Scalar = T;
    type Input = Image<C_IN, H, W, T>;
    type Output = Image<C_OUT, OH, OW, T>;
    type Params = (Kernels<C_OUT, C_IN, KH, KW, T>, [T; C_OUT]);

    fn forward(&self, (kernels, bias): &Self::Params, input: &Self::Input) -> Self::Output {
        let mut output = [[[T::ZERO; OW]; OH]; C_OUT];
        for (o, out_channel) in output.iter_mut().enumerate() {
            for (r, out_row) in out_channel.iter_mut().enumerate() {
                for (c, cell) in out_row.iter_mut().enumerate() {
                    let mut sum = bias[o];
                    for (i, plane) in input.iter().enumerate() {
                        for (p, kernel_row) in kernels[o][i].iter().enumerate() {
                            for (q, &weight) in kernel_row.iter().enumerate() {
                                sum += weight * plane[r + p][c + q];
                            }
                        }
                    }
                    *cell = sum;
                }
            }
        }
        output
    }

    /// Jede Ausgabestelle `(o, r, c)` mit Gradient `g` trägt bei:
    /// `∂L/∂b[o] += g`, `∂L/∂K[o][i][p][q] += g · x[i][r+p][c+q]`, `∂L/∂x[i][r+p][c+q] += K[o][i][p][q] · g`.
    fn backward(
        &self,
        (kernels, _): &Self::Params,
        (grad_kernels, grad_bias): &mut Self::Params,
        input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let mut grad_input = [[[T::ZERO; W]; H]; C_IN];
        for (o, grad_channel) in grad_output.iter().enumerate() {
            for (r, grad_row) in grad_channel.iter().enumerate() {
                for (c, &g) in grad_row.iter().enumerate() {
                    grad_bias[o] += g;
                    for (i, plane) in input.iter().enumerate() {
                        for p in 0..KH {
                            for q in 0..KW {
                                grad_kernels[o][i][p][q] += g * plane[r + p][c + q];
                                grad_input[i][r + p][c + q] += kernels[o][i][p][q] * g;
                            }
                        }
                    }
                }
            }
        }
        grad_input
    }

    /// Kerne aus dem `Initializer`, Bias 0. Jeder Ausgabewert sieht `C_IN·KH·KW` Eingaben
    /// (`fan_in`), jeder Eingabewert beeinflusst `C_OUT·KH·KW` Ausgaben (`fan_out`).
    fn init_params<I: Initializer>(&self, initializer: &I, rng: &mut Rng) -> Self::Params {
        let (fan_in, fan_out) = (C_IN * KH * KW, C_OUT * KH * KW);
        let mut kernels = <Kernels<C_OUT, C_IN, KH, KW, T> as Values<T>>::zeros();
        for index in 0..<Kernels<C_OUT, C_IN, KH, KW, T> as Values<T>>::LEN {
            *kernels.value_mut(index) = initializer.sample(rng, fan_in, fan_out);
        }
        (kernels, [T::ZERO; C_OUT])
    }
}

/// Eine Faltungsschicht.
pub type Conv2dLayer<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    T = f32,
> = Block<Conv2d<C_IN, C_OUT, H, W, KH, KW, OH, OW, T>>;

impl<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Block<Conv2d<C_IN, C_OUT, H, W, KH, KW, OH, OW, T>>
{
    pub fn new(kernels: Kernels<C_OUT, C_IN, KH, KW, T>, bias: [T; C_OUT]) -> Self {
        Self::from_parts(Conv2d::new(), (kernels, bias))
    }

    pub const fn kernels(&self) -> &Kernels<C_OUT, C_IN, KH, KW, T> {
        &self.params().0
    }

    pub const fn bias(&self) -> &[T; C_OUT] {
        &self.params().1
    }

    pub const fn grad_kernels(&self) -> &Kernels<C_OUT, C_IN, KH, KW, T> {
        &self.grads().0
    }

    pub const fn grad_bias(&self) -> &[T; C_OUT] {
        &self.grads().1
    }
}
