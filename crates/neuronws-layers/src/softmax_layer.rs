use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;

/// Macht aus `N` beliebigen Werten eine Wahrscheinlichkeitsverteilung: `y[i] = e^z[i] / Σ_j e^z[j]`.
///
/// Alle Ausgaben liegen in (0, 1) und summieren sich zu 1. Anders als eine elementweise
/// Aktivierung hängt jede Ausgabe von allen Eingaben ab, deshalb ist Softmax eine eigene Operation.
/// Braucht `exp`, also das Feature `std` oder `libm`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmax<const N: usize, T: Scalar = f32>(PhantomData<T>);

impl<const N: usize, T: Scalar> Softmax<N, T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const N: usize, T: Scalar> Operation for Softmax<N, T> {
    type Scalar = T;
    type Input = [T; N];
    type Output = [T; N];
    type Params = ();

    /// Vor dem Potenzieren wird der größte Wert abgezogen. Das ändert das Ergebnis nicht,
    /// verhindert aber, dass `exp` bei großen Eingaben überläuft.
    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        let max = input
            .iter()
            .copied()
            .fold(input[0], |max, z| if z > max { z } else { max });
        let exps = input.map(|z| (z - max).exp());
        let sum = exps.iter().fold(T::ZERO, |sum, &e| sum + e);
        exps.map(|e| e / sum)
    }

    /// `∂L/∂z[i] = y[i] · (g[i] − Σ_j g[j] · y[j])`
    ///
    /// Das ist die Jacobi-Matrix von Softmax, angewandt auf `g`, ohne dass sie je als Matrix
    /// aufgebaut wird.
    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &Self::Input,
        output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        let weighted = grad_output
            .iter()
            .zip(output)
            .fold(T::ZERO, |sum, (&g, &y)| sum + g * y);
        core::array::from_fn(|i| output[i] * (grad_output[i] - weighted))
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Softmax als Schicht.
pub type SoftmaxLayer<const N: usize, T = f32> = Block<Softmax<N, T>>;

impl<const N: usize, T: Scalar> Block<Softmax<N, T>> {
    pub fn new() -> Self {
        Self::from_parts(Softmax::new(), ())
    }
}

impl<const N: usize, T: Scalar> Default for Block<Softmax<N, T>> {
    fn default() -> Self {
        Self::new()
    }
}
