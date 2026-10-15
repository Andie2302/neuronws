use crate::activation::Activation;
use crate::traits::{
    BackwardData, BackwardPathMut, Data, ForwardData, ForwardPath, ForwardPathMut, Layer,
};

/// Wendet eine Aktivierungsfunktion elementweise auf `N` Werte an.
pub struct ActivationLayer<const N: usize, A: Activation> {
    activation: A,
    /// Die Eingabe `z` des letzten `forward_mut`, die Ableitung `φ'(z)` braucht sie.
    last_input: [f32; N],
    output: [f32; N],
    grad_input: [f32; N],
}

impl<const N: usize, A: Activation> ActivationLayer<N, A> {
    pub const fn new(activation: A) -> Self {
        Self {
            activation,
            last_input: [0.0; N],
            output: [0.0; N],
            grad_input: [0.0; N],
        }
    }
}

impl<const N: usize, A: Activation> Layer for ActivationLayer<N, A> {}

impl<const N: usize, A: Activation> Data for ActivationLayer<N, A> {}

impl<const N: usize, A: Activation> ForwardData for ActivationLayer<N, A> {
    type ForwardInput = [f32; N];
    type ForwardOutput = [f32; N];
}

impl<const N: usize, A: Activation> ForwardPath for ActivationLayer<N, A> {
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        input.map(|z| self.activation.apply(z))
    }
}

impl<const N: usize, A: Activation> ForwardPathMut for ActivationLayer<N, A> {
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        self.last_input = *input;
        self.output = self.forward(input);
        &self.output
    }
}

impl<const N: usize, A: Activation> BackwardData for ActivationLayer<N, A> {
    type BackwardInput = [f32; N];
    type BackwardOutput = [f32; N];
}

impl<const N: usize, A: Activation> BackwardPathMut for ActivationLayer<N, A> {
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
