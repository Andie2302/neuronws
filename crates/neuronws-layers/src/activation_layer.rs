use crate::activation::Activation;
use crate::traits::{Data, ForwardData, ForwardPath, Layer};

/// Wendet eine Aktivierungsfunktion elementweise auf `N` Werte an.
pub struct ActivationLayer<const N: usize, A: Activation> {
    activation: A,
}

impl<const N: usize, A: Activation> ActivationLayer<N, A> {
    pub const fn new(activation: A) -> Self {
        Self { activation }
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
