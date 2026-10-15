//! Faltung über Bilder: [`Conv2d`] mit Schrittweite, Rand und Dilation.
//!
//! # Die Geometrie
//!
//! Ein kleiner Kern der Größe `KH × KW` wird über das Bild geschoben. Wie er das tut, legen drei
//! Zahlen fest:
//!
//! | Zahl | Bedeutung | Standard |
//! |---|---|---|
//! | `STRIDE` | Schrittweite: um wie viele Pixel der Kern pro Schritt weiterrückt | 1 |
//! | `PAD` | Rand: so viele Nullen werden rundherum an das Bild angelegt | 0 |
//! | `DIL` | Dilation: Abstand zwischen den Kernpunkten (1 = dicht beieinander) | 1 |
//!
//! Die Ausgabegröße ergibt sich daraus (die Division rundet ab):
//!
//! ```text
//! OH = (H + 2·PAD − DIL·(KH − 1) − 1) / STRIDE + 1
//! ```
//!
//! Mit `PAD = DIL·(K − 1)/2` und Schrittweite 1 bleibt das Bild gleich groß ("same"), siehe
//! [`same_padding`].
//!
//! # Warum so viele Zahlen im Typ stehen
//!
//! Die Größen der Arrays gehören zum Typ, und Rust kann `H + 2·PAD − …` in einem Typ noch nicht
//! selbst ausrechnen (dazu bräuchte es `generic_const_exprs`, das noch instabil ist). Deshalb
//! stehen `OH` und `OW` als eigene Zahlen da. Zwei Hilfen nehmen die Arbeit ab:
//!
//! - [`conv_output_size`] rechnet die Ausgabegröße in einer `const`, und
//! - das Makro [`conv2d!`](crate::conv2d) schreibt den ganzen Typ mit allen elf Zahlen hin.
//!
//! Ein Typ, der alle Zahlen als Bündel trägt (etwa ein Trait `Geometry` mit zugeordneten
//! Konstanten), würde nicht helfen: `[T; G::H]` als Array-Länge ist auf stabilem Rust ebenfalls
//! nicht erlaubt.
//!
//! # Wann Größenfehler auffallen
//!
//! Stimmt `OH` nicht, bricht das Kompilieren ab, aber erst beim Bauen (`cargo build`,
//! `cargo test`), nicht bei `cargo check` und nicht in der Anzeige der IDE. Der Grund: Die Prüfung
//! steckt in `new()` und läuft erst, wenn der Compiler den konkreten Typ einsetzt. Mit
//! [`check_sizes!`](crate::check_sizes) lässt sich die Prüfung schon bei `cargo check` auslösen.

use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::{Image, Values};

/// Die Kerne einer Faltung: `kernels[ausgabekanal][eingabekanal][zeile][spalte]`.
pub type Kernels<const C_OUT: usize, const C_IN: usize, const KH: usize, const KW: usize, T = f32> =
    [[[[T; KW]; KH]; C_IN]; C_OUT];

/// Die Ausgabegröße einer Faltung entlang einer Achse:
/// `(input + 2·padding − dilation·(kernel − 1) − 1) / stride + 1`.
///
/// Als `const fn` benutzbar, um die Zahlen `OH` und `OW` nicht von Hand auszurechnen:
///
/// ```
/// use neuronws_layers::conv::{Conv2dLayer, conv_output_size};
///
/// // 28×28-Bild, 5×5-Kern, Schrittweite 2, Rand 2
/// const OUT: usize = conv_output_size(28, 5, 2, 2, 1);
/// assert_eq!(OUT, 14);
/// let _layer = Conv2dLayer::<1, 8, 28, 28, 5, 5, 2, 2, 1, OUT, OUT>::new(
///     [[[[0.0; 5]; 5]]; 8],
///     [0.0; 8],
/// );
/// ```
///
/// # Panics
/// Wenn `kernel`, `stride` oder `dilation` 0 sind oder der Kern (mit Dilation) größer ist als
/// das Bild mit Rand. In einer `const` ist das ein Compile-Fehler.
pub const fn conv_output_size(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    assert!(
        kernel >= 1 && stride >= 1 && dilation >= 1,
        "Faltung: Kern, Schrittweite und Dilation müssen mindestens 1 sein"
    );
    let reach = dilation * (kernel - 1) + 1;
    assert!(
        reach <= input + 2 * padding,
        "Faltung: der Kern (mit Dilation) ist größer als das Bild mit Rand"
    );
    (input + 2 * padding - reach) / stride + 1
}

/// Die Ausgabegröße einer **transponierten** Faltung entlang einer Achse:
/// `(input − 1)·stride + dilation·(kernel − 1) + 1 − 2·padding`.
///
/// Das ist die Umkehrung von [`conv_output_size`] (für den Fall, dass die Division dort aufgeht).
///
/// # Panics
/// Wenn `kernel`, `stride` oder `dilation` 0 sind, `input` 0 ist oder der Rand die Ausgabe
/// aufbrauchen würde.
pub const fn conv_transpose_output_size(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    assert!(
        input >= 1 && kernel >= 1 && stride >= 1 && dilation >= 1,
        "Transponierte Faltung: Eingabe, Kern, Schrittweite und Dilation müssen mindestens 1 sein"
    );
    let full = (input - 1) * stride + dilation * (kernel - 1) + 1;
    assert!(
        full > 2 * padding,
        "Transponierte Faltung: der Rand ist größer als die Ausgabe"
    );
    full - 2 * padding
}

/// Der Rand, bei dem eine Faltung mit Schrittweite 1 das Bild gleich groß lässt ("same"):
/// `dilation · (kernel − 1) / 2`.
///
/// # Panics
/// Wenn `dilation · (kernel − 1)` ungerade ist (bei geradem Kern ohne Dilation): Dann lässt sich
/// der Rand nicht gleichmäßig auf beide Seiten verteilen.
pub const fn same_padding(kernel: usize, dilation: usize) -> usize {
    assert!(
        kernel >= 1 && dilation >= 1,
        "same_padding: Kern und Dilation müssen mindestens 1 sein"
    );
    let reach = dilation * (kernel - 1);
    assert!(
        reach.is_multiple_of(2),
        "same_padding: bei geradem Kern lässt sich der Rand nicht gleichmäßig verteilen"
    );
    reach / 2
}

/// Das Pixel im Bild, auf das der Kernpunkt `k` bei Ausgabestelle `out` fällt, oder `None`, wenn
/// er auf den Rand (die angelegten Nullen) trifft.
///
/// Im Bild mit Rand liegt er bei `out · stride + k · dilation`, im echten Bild um `padding`
/// weniger.
#[inline]
pub(crate) const fn tap(
    out: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    size: usize,
) -> Option<usize> {
    let padded = out * stride + k * dilation;
    if padded < padding {
        return None;
    }
    let index = padded - padding;
    if index < size { Some(index) } else { None }
}

/// Die Ausgabestellen `first..end`, bei denen der Kernpunkt `k` im Bild liegt (und nicht im Rand).
///
/// Für diese Stellen `c` ist `c · stride + k · dilation − padding` ein gültiger Index in `0..size`.
/// So muss die innere Schleife nicht bei jedem Schritt auf den Rand prüfen und läuft über ein
/// zusammenhängendes Stück, das der Compiler gut in Vektorbefehle übersetzen kann.
#[inline]
pub(crate) const fn valid_outputs(
    k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    size: usize,
    outputs: usize,
) -> (usize, usize) {
    let offset = k * dilation;
    // erste Stelle, bei der `c · stride + offset ≥ padding`
    let first = if offset >= padding {
        0
    } else {
        (padding - offset).div_ceil(stride)
    };
    // eine hinter der letzten Stelle, bei der `c · stride + offset − padding ≤ size − 1`
    let end = if size + padding > offset {
        let last = (size - 1 + padding - offset) / stride;
        if last + 1 < outputs {
            last + 1
        } else {
            outputs
        }
    } else {
        0
    };
    if end < first {
        (first, first)
    } else {
        (first, end)
    }
}

/// `out[c] += weight · source[c · STRIDE]` für jede Stelle von `out`. Bei Schrittweite 1 sind
/// beide Streifen zusammenhängend, das ist die Form, die der Compiler vektorisiert.
#[inline(always)]
fn add_scaled<T: Scalar, const STRIDE: usize>(out: &mut [T], weight: T, source: &[T]) {
    if STRIDE == 1 {
        for (o, &x) in out.iter_mut().zip(source) {
            *o += weight * x;
        }
    } else {
        for (o, &x) in out.iter_mut().zip(source.iter().step_by(STRIDE)) {
            *o += weight * x;
        }
    }
}

/// `Σ_c grads[c] · source[c · STRIDE]`.
///
/// Bei Schrittweite 1 wird in acht getrennten Teilsummen gerechnet, die erst am Ende
/// zusammenkommen. Eine einzige laufende Summe zwingt die Rechnung, bei jeder Addition auf die
/// vorige zu warten. Mit acht Teilsummen laufen die Additionen nebeneinander (und als
/// Vektorbefehle). Die Reihenfolge der Rundungen ändert sich dadurch leicht, das Ergebnis ist
/// nicht schlechter, nur anders gerundet. Bei größerer Schrittweite bringt das nichts (die Werte
/// liegen nicht nebeneinander), dort wird einfach der Reihe nach summiert.
#[inline(always)]
fn scaled_sum<T: Scalar, const STRIDE: usize>(grads: &[T], source: &[T]) -> T {
    if STRIDE != 1 {
        return grads
            .iter()
            .zip(source.iter().step_by(STRIDE))
            .fold(T::ZERO, |sum, (&g, &x)| sum + g * x);
    }

    const LANES: usize = 8;
    let mut partial = [T::ZERO; LANES];
    // `as_chunks` zerlegt in Stücke aus genau `LANES` Werten und einen Rest. `source` ist meist
    // länger als `grads` (es reicht bis zum Zeilenende), zusammengehören nur die ersten
    // `grads.len()` Werte. Das `zip` hört beim kürzeren auf, und der Rest von `grads` bekommt
    // die Werte von `source` direkt dahinter.
    let (grad_chunks, grad_rest) = grads.as_chunks::<LANES>();
    let (source_chunks, _) = source.as_chunks::<LANES>();
    for (g_chunk, x_chunk) in grad_chunks.iter().zip(source_chunks) {
        for lane in 0..LANES {
            partial[lane] += g_chunk[lane] * x_chunk[lane];
        }
    }
    let mut sum = partial.iter().fold(T::ZERO, |sum, &p| sum + p);
    let full = grad_chunks.len() * LANES;
    for (&g, &x) in grad_rest.iter().zip(&source[full..]) {
        sum += g * x;
    }
    sum
}

/// `target[c · STRIDE] += weight · grads[c]` für jede Stelle von `grads`.
#[inline(always)]
fn add_scaled_to_strided<T: Scalar, const STRIDE: usize>(target: &mut [T], weight: T, grads: &[T]) {
    if STRIDE == 1 {
        for (t, &g) in target.iter_mut().zip(grads) {
            *t += weight * g;
        }
    } else {
        for (t, &g) in target.iter_mut().step_by(STRIDE).zip(grads) {
            *t += weight * g;
        }
    }
}

/// Zweidimensionale Faltung.
///
/// An jeder Stelle des Bildes wird die Summe aus Kern mal darunterliegenden Bildwerten (über alle
/// Eingabekanäle) plus Bias gebildet. Werte außerhalb des Bildes (im Rand) zählen als 0:
///
/// ```text
/// y[o][r][c] = b[o] + Σ_{i,p,q} K[o][i][p][q] · x[i][r·STRIDE + p·DIL − PAD][c·STRIDE + q·DIL − PAD]
/// ```
///
/// Dieselben Kernwerte gelten an jeder Stelle. Das spart Parameter und erkennt ein Muster überall
/// im Bild. Beim Zurückrechnen werden die Beiträge aller Stellen zu den Kernwerten aufsummiert.
/// Die Geometrie (`STRIDE`, `PAD`, `DIL`, `OH`, `OW`) ist im Modul-Text erklärt.
///
/// Weil die elf Zahlen unhandlich sind, gibt es das Makro [`conv2d!`](crate::conv2d), das
/// `OH` und `OW` ausrechnet, und für den einfachen Fall den Namen [`Conv2dValid`].
///
/// ```
/// use neuronws_layers::conv::Conv2dValidLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// // Ein Kanal rein, ein Kanal raus, 3×3-Bild, 2×2-Kern, 2×2 heraus.
/// let kernel = [[[[1.0, 0.0], [0.0, -1.0]]]];
/// let conv = Conv2dValidLayer::<1, 1, 3, 3, 2, 2, 2, 2>::new(kernel, [0.0]);
/// let image = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]];
/// assert_eq!(conv.forward(&image), [[[-4.0, -4.0], [-4.0, -4.0]]]);
/// ```
///
/// Stimmt die Ausgabegröße nicht, bricht das Kompilieren ab:
///
/// ```compile_fail
/// use neuronws_layers::conv::Conv2dValid;
///
/// // 5×5-Bild mit 3×3-Kern ergibt 3×3, nicht 4×4.
/// let _ = Conv2dValid::<1, 1, 5, 5, 3, 3, 4, 4>::new();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Conv2d<
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

/// Die einfache Faltung ohne Rand, mit Schrittweite 1 und ohne Dilation ("valid").
///
/// Die Ausgabegröße ist `OH = H − KH + 1` und `OW = W − KW + 1`.
pub type Conv2dValid<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    T = f32,
> = Conv2d<C_IN, C_OUT, H, W, KH, KW, 1, 0, 1, OH, OW, T>;

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
> Conv2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>
{
    /// # Compile-Fehler
    /// Wenn Kern, Schrittweite oder Dilation 0 sind, der Kern (mit Dilation) größer als das Bild
    /// mit Rand ist oder `OH`/`OW` nicht zur Formel im Modul-Text passen.
    pub const fn new() -> Self {
        const {
            assert!(
                OH == conv_output_size(H, KH, STRIDE, PAD, DIL)
                    && OW == conv_output_size(W, KW, STRIDE, PAD, DIL),
                "Conv2d: OH und OW müssen (H + 2·PAD − DIL·(KH − 1) − 1) / STRIDE + 1 und das \
                 Entsprechende für die Breite sein, siehe conv_output_size"
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
> Default for Conv2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>
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
> Operation for Conv2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>
{
    type Scalar = T;
    type Input = Image<C_IN, H, W, T>;
    type Output = Image<C_OUT, OH, OW, T>;
    type Params = (Kernels<C_OUT, C_IN, KH, KW, T>, [T; C_OUT]);

    /// Die Schleifen laufen **über die Kernpunkte außen** und über die Ausgabestellen innen: Für jedes
    /// Gewicht wird ein ganzer Streifen der Ausgabezeile in einem Zug aktualisiert. Jede einzelne
    /// Ausgabe summiert ihre Beiträge dabei in derselben Reihenfolge wie die einfache Schleife
    /// über `(i, p, q)`, das Ergebnis ist also dasselbe, nur schneller.
    fn forward(&self, (kernels, bias): &Self::Params, input: &Self::Input) -> Self::Output {
        let mut output = [[[T::ZERO; OW]; OH]; C_OUT];
        for (o, out_channel) in output.iter_mut().enumerate() {
            for row in out_channel.iter_mut() {
                row.fill(bias[o]);
            }
            for (i, plane) in input.iter().enumerate() {
                for (p, kernel_row) in kernels[o][i].iter().enumerate() {
                    for (q, &weight) in kernel_row.iter().enumerate() {
                        let (first, end) = valid_outputs(q, STRIDE, DIL, PAD, W, OW);
                        let start = first * STRIDE + q * DIL - PAD;
                        for (r, out_row) in out_channel.iter_mut().enumerate() {
                            let Some(source_row) = tap(r, p, STRIDE, DIL, PAD, H) else {
                                continue;
                            };
                            add_scaled::<T, STRIDE>(
                                &mut out_row[first..end],
                                weight,
                                &plane[source_row][start..],
                            );
                        }
                    }
                }
            }
        }
        output
    }

    /// Jede Ausgabestelle `(o, r, c)` mit Gradient `g` trägt über jeden Kernpunkt `(i, p, q)`, der
    /// im Bild liegt (Pixel `(y, x)`), bei:
    /// `∂L/∂b[o] += g`, `∂L/∂K[o][i][p][q] += g · x[i][y][x]`, `∂L/∂x[i][y][x] += K[o][i][p][q] · g`.
    ///
    /// Es gibt zwei Schleifenanordnungen mit demselben Ergebnis (bis auf die Rundung der letzten
    /// Stelle). Bei **breiten** Ausgabezeilen läuft die Rechnung wie vorwärts über die Kernpunkte
    /// außen und ganze Streifen innen. Bei **schmalen** Zeilen ist der Aufwand pro Streifen größer
    /// als sein Nutzen, dort ist die einfache Schleife über jede Ausgabestelle schneller. Welche
    /// gilt, steht beim Kompilieren fest (`OW` ist eine Konstante). Gemessen wurde mit
    /// `examples/bench_conv.rs`.
    fn backward(
        &self,
        (kernels, _): &Self::Params,
        (grad_kernels, grad_bias): &mut Self::Params,
        input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        /// Ab so vielen Ausgabespalten gilt die Anordnung "Kernpunkte außen".
        const WIDE_ROWS: usize = 16;

        for (o, grad_channel) in grad_output.iter().enumerate() {
            for row in grad_channel {
                for &g in row {
                    grad_bias[o] += g;
                }
            }
        }

        let mut grad_input = [[[T::ZERO; W]; H]; C_IN];
        if OW >= WIDE_ROWS {
            for (o, grad_channel) in grad_output.iter().enumerate() {
                for (i, plane) in input.iter().enumerate() {
                    for p in 0..KH {
                        for q in 0..KW {
                            let weight = kernels[o][i][p][q];
                            let (first, end) = valid_outputs(q, STRIDE, DIL, PAD, W, OW);
                            let start = first * STRIDE + q * DIL - PAD;
                            let mut kernel_gradient = grad_kernels[o][i][p][q];
                            for (r, grad_row) in grad_channel.iter().enumerate() {
                                let Some(source_row) = tap(r, p, STRIDE, DIL, PAD, H) else {
                                    continue;
                                };
                                let grads = &grad_row[first..end];
                                kernel_gradient +=
                                    scaled_sum::<T, STRIDE>(grads, &plane[source_row][start..]);
                                add_scaled_to_strided::<T, STRIDE>(
                                    &mut grad_input[i][source_row][start..],
                                    weight,
                                    grads,
                                );
                            }
                            grad_kernels[o][i][p][q] = kernel_gradient;
                        }
                    }
                }
            }
        } else {
            for (o, grad_channel) in grad_output.iter().enumerate() {
                for (r, grad_row) in grad_channel.iter().enumerate() {
                    for (c, &g) in grad_row.iter().enumerate() {
                        for (i, plane) in input.iter().enumerate() {
                            for p in 0..KH {
                                let Some(row) = tap(r, p, STRIDE, DIL, PAD, H) else {
                                    continue;
                                };
                                for q in 0..KW {
                                    if let Some(col) = tap(c, q, STRIDE, DIL, PAD, W) {
                                        grad_kernels[o][i][p][q] += g * plane[row][col];
                                        grad_input[i][row][col] += kernels[o][i][p][q] * g;
                                    }
                                }
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
        initializer.fill(rng, &mut kernels, fan_in, fan_out);
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
    const STRIDE: usize,
    const PAD: usize,
    const DIL: usize,
    const OH: usize,
    const OW: usize,
    T = f32,
> = Block<Conv2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>>;

/// Eine einfache Faltungsschicht ohne Rand, mit Schrittweite 1 und ohne Dilation.
pub type Conv2dValidLayer<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    T = f32,
> = Block<Conv2dValid<C_IN, C_OUT, H, W, KH, KW, OH, OW, T>>;

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
> Block<Conv2d<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, DIL, OH, OW, T>>
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

/// Schreibt den Typ einer [`Conv2d`] mit allen Zahlen hin und rechnet `OH` und `OW` selbst aus.
///
/// ```
/// use neuronws_layers::block::Block;
/// use neuronws_layers::init::Zeros;
/// use neuronws_layers::random::Rng;
/// use neuronws_layers::traits::ForwardPath;
///
/// // 1 Kanal rein, 4 raus, 28×28-Bild, 5×5-Kern, Schrittweite 2, Rand 2: ergibt 14×14.
/// type Layer = Block<neuronws_layers::conv2d!(in: 1, out: 4, size: (28, 28), kernel: (5, 5), stride: 2, padding: 2)>;
/// let layer = Layer::initialized(
///     <neuronws_layers::conv2d!(in: 1, out: 4, size: (28, 28), kernel: (5, 5), stride: 2, padding: 2)>::new(),
///     &Zeros,
///     &mut Rng::new(1),
/// );
/// let output = layer.forward(&[[[0.0; 28]; 28]]);
/// assert_eq!((output.len(), output[0].len(), output[0][0].len()), (4, 14, 14));
/// ```
///
/// Die Angaben stehen immer in dieser Reihenfolge, nur `stride`, `padding` und `dilation` dürfen
/// von hinten weggelassen werden (Standard: Schrittweite 1, Rand 0, Dilation 1):
///
/// - `conv2d!(in: C, out: K, size: (H, W), kernel: (KH, KW))`
/// - `conv2d!(…, stride: S)`
/// - `conv2d!(…, stride: S, padding: P)`
/// - `conv2d!(…, stride: S, padding: P, dilation: D)`
///
/// Der Typ ist die `Operation`. Für die ganze Schicht gibt es [`conv2d_layer!`](crate::conv2d_layer).
/// Zahlen müssen Literale oder Konstanten sein, keine Typ-Parameter (dort kann Rust nicht rechnen).
#[macro_export]
macro_rules! conv2d {
    (in: $c_in:expr, out: $c_out:expr, size: ($h:expr, $w:expr), kernel: ($kh:expr, $kw:expr)) => {
        $crate::conv2d!(in: $c_in, out: $c_out, size: ($h, $w), kernel: ($kh, $kw), stride: 1, padding: 0, dilation: 1)
    };
    (in: $c_in:expr, out: $c_out:expr, size: ($h:expr, $w:expr), kernel: ($kh:expr, $kw:expr), stride: $stride:expr) => {
        $crate::conv2d!(in: $c_in, out: $c_out, size: ($h, $w), kernel: ($kh, $kw), stride: $stride, padding: 0, dilation: 1)
    };
    (in: $c_in:expr, out: $c_out:expr, size: ($h:expr, $w:expr), kernel: ($kh:expr, $kw:expr), stride: $stride:expr, padding: $pad:expr) => {
        $crate::conv2d!(in: $c_in, out: $c_out, size: ($h, $w), kernel: ($kh, $kw), stride: $stride, padding: $pad, dilation: 1)
    };
    (in: $c_in:expr, out: $c_out:expr, size: ($h:expr, $w:expr), kernel: ($kh:expr, $kw:expr), stride: $stride:expr, padding: $pad:expr, dilation: $dil:expr) => {
        $crate::conv::Conv2d<
            { $c_in }, { $c_out }, { $h }, { $w }, { $kh }, { $kw },
            { $stride }, { $pad }, { $dil },
            { $crate::conv::conv_output_size($h, $kh, $stride, $pad, $dil) },
            { $crate::conv::conv_output_size($w, $kw, $stride, $pad, $dil) },
        >
    };
}

/// Wie [`conv2d!`](crate::conv2d), aber für die ganze Schicht (`Block<Conv2d<…>>`).
#[macro_export]
macro_rules! conv2d_layer {
    ($($arguments:tt)*) => {
        $crate::block::Block<$crate::conv2d!($($arguments)*)>
    };
}
