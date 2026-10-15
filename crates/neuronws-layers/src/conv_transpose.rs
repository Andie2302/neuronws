//! Transponierte Faltung zum Hochskalieren: [`ConvTranspose2d`].
//!
//! Eine normale Faltung mit Schrittweite 2 halbiert die Bildgröße. Die transponierte Faltung
//! (oft "Deconvolution" genannt) macht den umgekehrten Weg und **vergrößert**: Jeder Eingabewert
//! "stempelt" den ganzen Kern, mit sich selbst multipliziert, an seine Stelle in der Ausgabe, wobei
//! sich überlappende Stempel addieren. Sie steckt in Decodern von Autoencodern und in Netzen,
//! die Bilder erzeugen.
//!
//! Mathematisch ist sie die **Adjungierte** der Faltung mit derselben Geometrie: Sie rechnet
//! genau das, was beim Zurückrechnen einer Faltung mit dem Gradienten passiert. Deshalb gilt für
//! gleiche Kerne `⟨Faltung(x), y⟩ = ⟨x, TransponierteFaltung(y)⟩`.
//!
//! Die Kerne haben die Form `kernels[eingabekanal][ausgabekanal][zeile][spalte]`, also dieselbe
//! Anordnung wie bei der Faltung, deren Adjungierte sie ist. Die Ausgabegröße folgt aus
//! [`conv_transpose_output_size`].

use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::conv::{conv_transpose_output_size, tap};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::{Image, Values};

/// Die Kerne einer transponierten Faltung: `kernels[eingabekanal][ausgabekanal][zeile][spalte]`.
pub type TransposeKernels<
    const C_IN: usize,
    const C_OUT: usize,
    const KH: usize,
    const KW: usize,
    T = f32,
> = [[[[T; KW]; KH]; C_OUT]; C_IN];

/// Transponierte zweidimensionale Faltung:
///
/// ```text
/// y[o][r·STRIDE + p·DIL − PAD][c·STRIDE + q·DIL − PAD]  +=  x[i][r][c] · K[i][o][p][q]
/// ```
///
/// für alle Eingabestellen `(i, r, c)` und Kernpunkte `(o, p, q)`, die in der Ausgabe liegen; dazu
/// kommt der Bias `b[o]`. Die Ausgabegröße ist
/// `OH = (H − 1)·STRIDE + DIL·(KH − 1) + 1 − 2·PAD`, siehe [`conv_transpose_output_size`]. Sie steht
/// wie bei der Faltung als eigene Zahl da, auch dafür gibt es ein Makro:
/// [`conv_transpose2d!`](crate::conv_transpose2d).
///
/// ```
/// use neuronws_layers::conv_transpose::ConvTranspose2dLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// // 2×2 → 4×4 mit Schrittweite 2 und 2×2-Kern: Jeder Eingabewert füllt einen eigenen 2×2-Block.
/// let layer = ConvTranspose2dLayer::<1, 1, 2, 2, 2, 2, 2, 0, 1, 4, 4>::new([[[[1.0; 2]; 2]]], [0.0]);
/// let output = layer.forward(&[[[1.0, 2.0], [3.0, 4.0]]]);
/// assert_eq!(output[0][0], [1.0, 1.0, 2.0, 2.0]);
/// assert_eq!(output[0][3], [3.0, 3.0, 4.0, 4.0]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ConvTranspose2d<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
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
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> ConvTranspose2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>
{
    /// # Compile-Fehler
    /// Wenn Eingabe, Kern, Schrittweite oder Dilation 0 sind, der Rand die Ausgabe aufbraucht oder
    /// `OH`/`OW` nicht zur Formel passen.
    pub const fn new() -> Self {
        const {
            assert!(
                OH == conv_transpose_output_size(H, KH, STRIDE, PAD, DIL)
                    && OW == conv_transpose_output_size(W, KW, STRIDE, PAD, DIL),
                "ConvTranspose2d: OH und OW müssen (H − 1)·STRIDE + DIL·(KH − 1) + 1 − 2·PAD und \
                 das Entsprechende für die Breite sein, siehe conv_transpose_output_size"
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
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Default for ConvTranspose2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>
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
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Operation for ConvTranspose2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>
{
    type Scalar = T;
    type Input = Image<C_IN, H, W, T>;
    type Output = Image<C_OUT, OH, OW, T>;
    type Params = (TransposeKernels<C_IN, C_OUT, KH, KW, T>, [T; C_OUT]);

    fn forward(&self, (kernels, bias): &Self::Params, input: &Self::Input) -> Self::Output {
        let mut output = [[[T::ZERO; OW]; OH]; C_OUT];
        for (o, channel) in output.iter_mut().enumerate() {
            for row in channel.iter_mut() {
                row.fill(bias[o]);
            }
        }
        for (i, plane) in input.iter().enumerate() {
            for (r, row) in plane.iter().enumerate() {
                for (c, &x) in row.iter().enumerate() {
                    for (o, channel) in output.iter_mut().enumerate() {
                        for (p, kernel_row) in kernels[i][o].iter().enumerate() {
                            let Some(out_row) = tap(r, p, STRIDE, DIL, PAD, OH) else {
                                continue;
                            };
                            for (q, &weight) in kernel_row.iter().enumerate() {
                                if let Some(out_col) = tap(c, q, STRIDE, DIL, PAD, OW) {
                                    channel[out_row][out_col] += x * weight;
                                }
                            }
                        }
                    }
                }
            }
        }
        output
    }

    /// Mit dem Gradienten `g` an der Ausgabestelle, auf die ein Stempel `(i, r, c, o, p, q)`
    /// fiel: `∂L/∂K[i][o][p][q] += x[i][r][c] · g`, `∂L/∂x[i][r][c] += K[i][o][p][q] · g`.
    /// Der Bias bekommt die Summe aller Gradienten seines Kanals.
    fn backward(
        &self,
        (kernels, _): &Self::Params,
        (grad_kernels, grad_bias): &mut Self::Params,
        input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        for (o, channel) in grad_output.iter().enumerate() {
            for row in channel {
                for &g in row {
                    grad_bias[o] += g;
                }
            }
        }

        let mut grad_input = [[[T::ZERO; W]; H]; C_IN];
        for (i, plane) in input.iter().enumerate() {
            for (r, row) in plane.iter().enumerate() {
                for (c, &x) in row.iter().enumerate() {
                    for (o, channel) in grad_output.iter().enumerate() {
                        for (p, kernel_row) in kernels[i][o].iter().enumerate() {
                            let Some(out_row) = tap(r, p, STRIDE, DIL, PAD, OH) else {
                                continue;
                            };
                            for (q, &weight) in kernel_row.iter().enumerate() {
                                if let Some(out_col) = tap(c, q, STRIDE, DIL, PAD, OW) {
                                    let g = channel[out_row][out_col];
                                    grad_kernels[i][o][p][q] += x * g;
                                    grad_input[i][r][c] += weight * g;
                                }
                            }
                        }
                    }
                }
            }
        }
        grad_input
    }

    /// Kerne aus dem `Initializer`, Bias 0. Jeder Ausgabewert sieht `C_IN·KH·KW` Stempel
    /// (`fan_in`), jeder Eingabewert beeinflusst `C_OUT·KH·KW` Ausgaben (`fan_out`).
    fn init_params<I: Initializer>(&self, initializer: &I, rng: &mut Rng) -> Self::Params {
        let mut kernels = <TransposeKernels<C_IN, C_OUT, KH, KW, T> as Values<T>>::zeros();
        initializer.fill(rng, &mut kernels, C_IN * KH * KW, C_OUT * KH * KW);
        (kernels, [T::ZERO; C_OUT])
    }
}

/// Eine transponierte Faltungsschicht.
pub type ConvTranspose2dLayer<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OH: usize,
    const OW: usize,
    T = f32,
> = Block<ConvTranspose2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>>;

impl<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Block<ConvTranspose2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>>
{
    pub fn new(kernels: TransposeKernels<C_IN, C_OUT, KH, KW, T>, bias: [T; C_OUT]) -> Self {
        Self::from_parts(ConvTranspose2d::new(), (kernels, bias))
    }

    pub const fn kernels(&self) -> &TransposeKernels<C_IN, C_OUT, KH, KW, T> {
        &self.params().0
    }

    pub const fn bias(&self) -> &[T; C_OUT] {
        &self.params().1
    }

    pub const fn grad_kernels(&self) -> &TransposeKernels<C_IN, C_OUT, KH, KW, T> {
        &self.grads().0
    }

    pub const fn grad_bias(&self) -> &[T; C_OUT] {
        &self.grads().1
    }
}

/// Schreibt den Typ einer [`ConvTranspose2d`] mit allen Zahlen hin und rechnet `OH` und `OW`
/// selbst aus.
///
/// ```
/// use neuronws_layers::block::Block;
///
/// // 4 Kanäle 7×7 → 1 Kanal 14×14? Mit Schrittweite 2, Kern 2 und Rand 0 ergibt (7−1)·2+2 = 14.
/// type Decoder = Block<neuronws_layers::conv_transpose2d!(in: 4, out: 1, size: (7, 7), kernel: (2, 2), stride: 2)>;
/// let decoder = Decoder::new([[[[0.0; 2]; 2]; 1]; 4], [0.0]);
/// # let _ = decoder;
/// ```
///
/// Die Angaben stehen in dieser Reihenfolge, `stride`, `padding` und `dilation` dürfen von hinten
/// weggelassen werden (Standard: Schrittweite 1, Rand 0, Dilation 1), wie bei
/// [`conv2d!`](crate::conv2d).
#[macro_export]
macro_rules! conv_transpose2d {
    (in: $c_in:expr, out: $c_out:expr, size: ($h:expr, $w:expr), kernel: ($kh:expr, $kw:expr)) => {
        $crate::conv_transpose2d!(in: $c_in, out: $c_out, size: ($h, $w), kernel: ($kh, $kw), stride: 1, padding: 0, dilation: 1)
    };
    (in: $c_in:expr, out: $c_out:expr, size: ($h:expr, $w:expr), kernel: ($kh:expr, $kw:expr), stride: $stride:expr) => {
        $crate::conv_transpose2d!(in: $c_in, out: $c_out, size: ($h, $w), kernel: ($kh, $kw), stride: $stride, padding: 0, dilation: 1)
    };
    (in: $c_in:expr, out: $c_out:expr, size: ($h:expr, $w:expr), kernel: ($kh:expr, $kw:expr), stride: $stride:expr, padding: $pad:expr) => {
        $crate::conv_transpose2d!(in: $c_in, out: $c_out, size: ($h, $w), kernel: ($kh, $kw), stride: $stride, padding: $pad, dilation: 1)
    };
    (in: $c_in:expr, out: $c_out:expr, size: ($h:expr, $w:expr), kernel: ($kh:expr, $kw:expr), stride: $stride:expr, padding: $pad:expr, dilation: $dil:expr) => {
        $crate::conv_transpose::ConvTranspose2d<
            { $c_in }, { $c_out }, { $h }, { $w }, { $kh }, { $kw },
            { $stride }, { $pad }, { $dil },
            { $crate::conv::conv_transpose_output_size($h, $kh, $stride, $pad, $dil) },
            { $crate::conv::conv_transpose_output_size($w, $kw, $stride, $pad, $dil) },
        >
    };
}
