use crate::scalar::Scalar;
use crate::traits::{
    BackwardData, BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer, Trainable,
};

/// Macht aus `N` beliebigen Werten eine Wahrscheinlichkeitsverteilung: `y[i] = e^z[i] / Σ_j e^z[j]`.
///
/// Alle Ausgaben liegen in (0, 1) und summieren sich zu 1. Anders als eine elementweise
/// Aktivierung hängt jede Ausgabe von allen Eingaben ab, deshalb ist Softmax eine eigene Schicht.
/// Braucht `exp`, also das Feature `std` oder `libm`.
pub struct SoftmaxLayer<const N: usize, T: Scalar = f32> {
    output: [T; N],
    grad_input: [T; N],
}

impl<const N: usize, T: Scalar> SoftmaxLayer<N, T> {
    pub const fn new() -> Self {
        Self {
            output: [T::ZERO; N],
            grad_input: [T::ZERO; N],
        }
    }
}

impl<const N: usize, T: Scalar> Default for SoftmaxLayer<N, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, T: Scalar> Layer for SoftmaxLayer<N, T> {}

impl<const N: usize, T: Scalar> ForwardData for SoftmaxLayer<N, T> {
    type ForwardInput = [T; N];
    type ForwardOutput = [T; N];
}

impl<const N: usize, T: Scalar> ForwardPath for SoftmaxLayer<N, T> {
    /// Vor dem Potenzieren wird der größte Wert abgezogen. Das ändert das Ergebnis nicht,
    /// verhindert aber, dass `exp` bei großen Eingaben überläuft.
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        let max = input
            .iter()
            .copied()
            .fold(input[0], |max, z| if z > max { z } else { max });
        let exps = input.map(|z| (z - max).exp());
        let sum = exps.iter().fold(T::ZERO, |sum, &e| sum + e);
        exps.map(|e| e / sum)
    }
}

impl<const N: usize, T: Scalar> ForwardPathMut for SoftmaxLayer<N, T> {
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        self.output = self.forward(input);
        &self.output
    }
}

impl<const N: usize, T: Scalar> BackwardData for SoftmaxLayer<N, T> {
    type BackwardInput = [T; N];
    type BackwardOutput = [T; N];
}

impl<const N: usize, T: Scalar> BackwardPathMut for SoftmaxLayer<N, T> {
    /// `∂L/∂z[i] = y[i] · (g[i] − Σ_j g[j] · y[j])`
    ///
    /// Das ist die Jacobi-Matrix von Softmax, angewandt auf `g`, ohne dass sie je als Matrix
    /// aufgebaut wird.
    fn backward_mut(&mut self, grad_output: &Self::BackwardInput) -> &Self::BackwardOutput {
        let weighted = grad_output
            .iter()
            .zip(&self.output)
            .fold(T::ZERO, |sum, (&g, &y)| sum + g * y);
        for ((grad_z, &g), &y) in self
            .grad_input
            .iter_mut()
            .zip(grad_output)
            .zip(&self.output)
        {
            *grad_z = y * (g - weighted);
        }
        &self.grad_input
    }
}

impl<const N: usize, T: Scalar> Trainable for SoftmaxLayer<N, T> {
    type Scalar = T;

    const PARAMETER_COUNT: usize = 0;

    fn visit_parameters(&mut self, _visitor: &mut impl FnMut(&mut T, T)) {}

    fn zero_grad(&mut self) {}
}
