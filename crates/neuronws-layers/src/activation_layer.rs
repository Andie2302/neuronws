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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{Relu, Step};
    use crate::dense_layer::DenseLayer;

    #[test]
    fn relu_layer_clips_each_element() {
        let layer = ActivationLayer::<3, _>::new(Relu);
        assert_eq!(layer.forward(&[-1.0, 0.0, 2.0]), [0.0, 0.0, 2.0]);
    }

    #[test]
    fn dense_followed_by_step_is_a_perceptron() {
        // AND-Gatter: 1 nur, wenn beide Eingänge 1 sind
        let dense = DenseLayer::new([[1.0, 1.0]], [-1.5]);
        let step = ActivationLayer::<1, _>::new(Step);
        let and = |x: [f32; 2]| step.forward(&dense.forward(&x));

        assert_eq!(and([0.0, 0.0]), [0.0]);
        assert_eq!(and([1.0, 0.0]), [0.0]);
        assert_eq!(and([0.0, 1.0]), [0.0]);
        assert_eq!(and([1.0, 1.0]), [1.0]);
    }
}
