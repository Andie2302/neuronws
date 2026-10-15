use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::values::Image;

/// Max-Pooling: aus jedem `P × P`-Fenster (ohne Überlappung) bleibt nur der größte Wert.
///
/// Das Bild wird kleiner (`OH = H / P`, `OW = W / P`) und unempfindlicher gegen kleine
/// Verschiebungen. Ein Rest am Rand, der kein volles Fenster füllt, fällt weg.
/// Hat keine Parameter.
///
/// Beim Zurückrechnen bekommt nur die Stelle des Maximums den Gradienten, alle anderen im Fenster
/// 0. Sind mehrere Werte gleich groß, zählt der erste (zeilenweise).
///
/// Die Ausgabegröße steht als eigene Zahl da und wird beim Kompilieren geprüft (erst beim Bauen,
/// nicht bei `cargo check`, außer man schreibt sie zusätzlich in [`check_sizes!`](crate::check_sizes)):
///
/// ```compile_fail
/// use neuronws_layers::pooling::MaxPool2d;
///
/// // 4×4-Bild mit 2×2-Fenster ergibt 2×2, nicht 3×3.
/// let _ = MaxPool2d::<1, 4, 4, 2, 3, 3>::new();
/// ```
///
/// ```
/// use neuronws_layers::pooling::MaxPool2dLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// let pool = MaxPool2dLayer::<1, 4, 4, 2, 2, 2>::new();
/// let image = [[[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 5.0], [0.0, 0.0, 6.0, 0.0], [7.0, 0.0, 0.0, 0.0]]];
/// assert_eq!(pool.forward(&image), [[[4.0, 5.0], [7.0, 6.0]]]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MaxPool2d<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar = f32,
> {
    scalar: PhantomData<T>,
}

const fn check_pooling(h: usize, w: usize, p: usize, oh: usize, ow: usize) {
    assert!(p >= 1, "Pooling: das Fenster muss mindestens 1×1 sein");
    assert!(
        oh == h / p && ow == w / p && oh >= 1 && ow >= 1,
        "Pooling: die Ausgabegröße muss (H / P) × (W / P) sein und mindestens 1×1"
    );
}

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> MaxPool2d<C, H, W, P, OH, OW, T>
{
    /// # Compile-Fehler
    /// Wenn `OH`/`OW` nicht `H / P`/`W / P` sind.
    pub const fn new() -> Self {
        const { check_pooling(H, W, P, OH, OW) };
        Self {
            scalar: PhantomData,
        }
    }
}

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Default for MaxPool2d<C, H, W, P, OH, OW, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Operation for MaxPool2d<C, H, W, P, OH, OW, T>
{
    type Scalar = T;
    type Input = Image<C, H, W, T>;
    type Output = Image<C, OH, OW, T>;
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        let mut output = [[[T::ZERO; OW]; OH]; C];
        for (c, plane) in input.iter().enumerate() {
            for r in 0..OH {
                for s in 0..OW {
                    let (p, q) = argmax_in_window::<T, W, P>(plane, r * P, s * P);
                    output[c][r][s] = plane[r * P + p][s * P + q];
                }
            }
        }
        output
    }

    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let mut grad_input = [[[T::ZERO; W]; H]; C];
        for (c, plane) in input.iter().enumerate() {
            for r in 0..OH {
                for s in 0..OW {
                    let (p, q) = argmax_in_window::<T, W, P>(plane, r * P, s * P);
                    grad_input[c][r * P + p][s * P + q] += grad_output[c][r][s];
                }
            }
        }
        grad_input
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Position `(zeile, spalte)` des ersten größten Werts im `P × P`-Fenster ab `(top, left)`.
fn argmax_in_window<T: Scalar, const W: usize, const P: usize>(
    plane: &[[T; W]],
    top: usize,
    left: usize,
) -> (usize, usize) {
    let mut best = (0, 0);
    let mut best_value = plane[top][left];
    for p in 0..P {
        for q in 0..P {
            let value = plane[top + p][left + q];
            if value > best_value {
                best = (p, q);
                best_value = value;
            }
        }
    }
    best
}

/// Eine Max-Pooling-Schicht.
pub type MaxPool2dLayer<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T = f32,
> = Block<MaxPool2d<C, H, W, P, OH, OW, T>>;

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Block<MaxPool2d<C, H, W, P, OH, OW, T>>
{
    pub fn new() -> Self {
        Self::from_parts(MaxPool2d::new(), ())
    }
}

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Default for Block<MaxPool2d<C, H, W, P, OH, OW, T>>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Average-Pooling: aus jedem `P × P`-Fenster (ohne Überlappung) wird der Mittelwert.
///
/// Sonst wie [`MaxPool2d`]. Beim Zurückrechnen bekommt jede Stelle im Fenster den gleichen
/// Anteil `1 / P²` des Gradienten.
#[derive(Debug, Clone, Copy)]
pub struct AvgPool2d<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar = f32,
> {
    scalar: PhantomData<T>,
}

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> AvgPool2d<C, H, W, P, OH, OW, T>
{
    /// # Compile-Fehler
    /// Wenn `OH`/`OW` nicht `H / P`/`W / P` sind.
    pub const fn new() -> Self {
        const { check_pooling(H, W, P, OH, OW) };
        Self {
            scalar: PhantomData,
        }
    }
}

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Default for AvgPool2d<C, H, W, P, OH, OW, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Operation for AvgPool2d<C, H, W, P, OH, OW, T>
{
    type Scalar = T;
    type Input = Image<C, H, W, T>;
    type Output = Image<C, OH, OW, T>;
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        let area = T::from_usize(P * P);
        let mut output = [[[T::ZERO; OW]; OH]; C];
        for (c, plane) in input.iter().enumerate() {
            for r in 0..OH {
                for s in 0..OW {
                    let mut sum = T::ZERO;
                    for p in 0..P {
                        for q in 0..P {
                            sum += plane[r * P + p][s * P + q];
                        }
                    }
                    output[c][r][s] = sum / area;
                }
            }
        }
        output
    }

    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &Self::Input,
        _output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let area = T::from_usize(P * P);
        let mut grad_input = [[[T::ZERO; W]; H]; C];
        for (c, grad_plane) in grad_output.iter().enumerate() {
            for (r, grad_row) in grad_plane.iter().enumerate() {
                for (s, &g) in grad_row.iter().enumerate() {
                    for p in 0..P {
                        for q in 0..P {
                            grad_input[c][r * P + p][s * P + q] = g / area;
                        }
                    }
                }
            }
        }
        grad_input
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Eine Average-Pooling-Schicht.
pub type AvgPool2dLayer<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T = f32,
> = Block<AvgPool2d<C, H, W, P, OH, OW, T>>;

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Block<AvgPool2d<C, H, W, P, OH, OW, T>>
{
    pub fn new() -> Self {
        Self::from_parts(AvgPool2d::new(), ())
    }
}

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const P: usize,
    const OH: usize,
    const OW: usize,
    T: Scalar,
> Default for Block<AvgPool2d<C, H, W, P, OH, OW, T>>
{
    fn default() -> Self {
        Self::new()
    }
}
