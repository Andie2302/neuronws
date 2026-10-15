use crate::activation::Activation;
use crate::scalar::Scalar;
use crate::traits::{
    BackwardData, BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer, Trainable,
};

/// Wendet eine Aktivierungsfunktion elementweise auf `N` Werte an.
pub struct ActivationLayer<const N: usize, A: Activation, T: Scalar = f32> {
    activation: A,
    /// Die Eingabe `z` des letzten `forward_mut`, die Ableitung `φ'(z)` braucht sie.
    last_input: [T; N],
    output: [T; N],
    grad_input: [T; N],
}

impl<const N: usize, A: Activation, T: Scalar> ActivationLayer<N, A, T> {
    pub const fn new(activation: A) -> Self {
        Self {
            activation,
            last_input: [T::ZERO; N],
            output: [T::ZERO; N],
            grad_input: [T::ZERO; N],
        }
    }
}

impl<const N: usize, A: Activation, T: Scalar> Layer for ActivationLayer<N, A, T> {}

impl<const N: usize, A: Activation, T: Scalar> ForwardData for ActivationLayer<N, A, T> {
    type ForwardInput = [T; N];
    type ForwardOutput = [T; N];
}

impl<const N: usize, A: Activation, T: Scalar> ForwardPath for ActivationLayer<N, A, T> {
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        input.map(|z| self.activation.apply(z))
    }
}

impl<const N: usize, A: Activation, T: Scalar> ForwardPathMut for ActivationLayer<N, A, T> {
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        self.last_input = *input;
        self.output = self.forward(input);
        &self.output
    }
}

impl<const N: usize, A: Activation, T: Scalar> BackwardData for ActivationLayer<N, A, T> {
    type BackwardInput = [T; N];
    type BackwardOutput = [T; N];
}

impl<const N: usize, A: Activation, T: Scalar> BackwardPathMut for ActivationLayer<N, A, T> {
    /// Kettenregel elementweise: `∂L/∂z[i] = g[i] · φ'(z[i])`.
    fn backward_mut(&mut self, grad_output: &Self::BackwardInput) -> &Self::BackwardOutput {
        for ((grad_z, &g), &z) in self
            .grad_input
            .iter_mut()
            .zip(grad_output)
            .zip(&self.last_input)
        {
            *grad_z = g * self.activation.derivative(z);
        }
        &self.grad_input
    }
}

/// Die Aktivierungsschicht hat keine lernbaren Parameter, es gibt also nichts zu besuchen.
impl<const N: usize, A: Activation, T: Scalar> Trainable for ActivationLayer<N, A, T> {
    type Scalar = T;

    fn visit_parameters(&mut self, _visitor: &mut impl FnMut(&mut T, T)) {}

    fn zero_grad(&mut self) {}
}
