use crate::traits::{Data, ForwardData, ForwardPath, Layer};

pub struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: [[f32; IN]; OUT],
    bias: [f32; OUT],
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub const fn new(weights: [[f32; IN]; OUT], bias: [f32; OUT]) -> Self {
        Self { weights, bias }
    }
}

impl<const IN: usize, const OUT: usize> Layer for DenseLayer<IN, OUT> {}

impl<const IN: usize, const OUT: usize> Data for DenseLayer<IN, OUT> {}

impl<const IN: usize, const OUT: usize> ForwardData for DenseLayer<IN, OUT> {
    type ForwardInput = [f32; IN];
    type ForwardOutput = [f32; OUT];
}

impl<const IN: usize, const OUT: usize> ForwardPath for DenseLayer<IN, OUT> {
    /// y[j] = Σ_i W[j][i] * x[i] + b[j], noch ohne Aktivierung.
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        core::array::from_fn(|j| {
            let dot: f32 = self.weights[j].iter().zip(input).map(|(w, x)| w * x).sum();
            dot + self.bias[j]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_is_matrix_times_vector_plus_bias() {
        // 2 Eingänge -> 3 Neuronen
        let layer = DenseLayer::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [0.5, 0.0, -1.0]);
        assert_eq!(layer.forward(&[1.0, 10.0]), [21.5, 43.0, 64.0]);
    }

    #[test]
    fn one_output_behaves_like_a_single_perceptron() {
        let layer = DenseLayer::new([[2.0, -1.0]], [0.0]);
        assert_eq!(layer.forward(&[3.0, 4.0]), [2.0]);
    }
}
