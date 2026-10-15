//! Eine eigene Schicht, die es in der Bibliothek nicht gibt: skaliert und verschiebt jede Eingabe
//! einzeln, mit lernbarem Faktor `a` und Verschiebung `b` pro Eingabe: `y[i] = a[i] · x[i] + b[i]`.
//!
//! Das ist alles, was man dafür schreiben muss. Zwischenspeicher, Gradienten sammeln, alle Traits
//! und das Lernen liefert `Block`.
#![allow(dead_code)]

use core::marker::PhantomData;

use neuronws_layers::block::Operation;
use neuronws_layers::init::Initializer;
use neuronws_layers::random::Rng;
use neuronws_layers::scalar::Scalar;

#[derive(Debug, Clone, Copy, Default)]
pub struct ScaleShift<const N: usize, T: Scalar = f32>(PhantomData<T>);

impl<const N: usize, T: Scalar> ScaleShift<N, T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const N: usize, T: Scalar> Operation for ScaleShift<N, T> {
    type Scalar = T;
    type Input = [T; N];
    type Output = [T; N];
    type Params = ([T; N], [T; N]); // (a, b)

    fn forward(&self, (a, b): &Self::Params, x: &Self::Input) -> Self::Output {
        core::array::from_fn(|i| a[i] * x[i] + b[i])
    }

    fn backward(
        &self,
        (a, _): &Self::Params,
        (grad_a, grad_b): &mut Self::Params,
        x: &Self::Input,
        _y: &Self::Output,
        g: &Self::Output,
    ) -> Self::Input {
        for i in 0..N {
            grad_a[i] += g[i] * x[i];
            grad_b[i] += g[i];
        }
        core::array::from_fn(|i| g[i] * a[i])
    }

    /// Sinnvoller Start ist "nichts ändern": `a = 1`, `b = 0`, egal welche Startwerte gewählt sind.
    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) -> Self::Params {
        ([T::ONE; N], [T::ZERO; N])
    }
}

/// Eine absichtlich falsche Operation: `y = 2·x`, aber `backward` vergisst den Faktor 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct BrokenDouble;

impl Operation for BrokenDouble {
    type Scalar = f32;
    type Input = [f32; 2];
    type Output = [f32; 2];
    type Params = ();

    fn forward(&self, _: &(), x: &Self::Input) -> Self::Output {
        x.map(|v| 2.0 * v)
    }

    fn backward(
        &self,
        _: &(),
        _: &mut (),
        _x: &Self::Input,
        _y: &Self::Output,
        g: &Self::Output,
    ) -> Self::Input {
        *g // richtig wäre g.map(|v| 2.0 * v)
    }

    fn init_params<I: Initializer>(&self, _: &I, _: &mut Rng) {}
}
