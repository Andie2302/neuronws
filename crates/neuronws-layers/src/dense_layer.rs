use crate::traits::{Data, ForwardData, ForwardPath, Layer};

pub struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: [[f32; IN]; OUT],
    bias: [f32; OUT],
    /// Die Eingabe des letzten `forward_mut`, der Backward-Pass braucht sie für `∂L/∂W`.
    last_input: [f32; IN],
    /// Aufsummierte Gradienten, bis `zero_grad` sie zurücksetzt.
    grad_weights: [[f32; IN]; OUT],
    grad_bias: [f32; OUT],
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub const fn new(weights: [[f32; IN]; OUT], bias: [f32; OUT]) -> Self {
        Self {
            weights,
            bias,
            last_input: [0.0; IN],
            grad_weights: [[0.0; IN]; OUT],
            grad_bias: [0.0; OUT],
        }
    }

    /// Wie `forward`, merkt sich aber die Eingabe für den Backward-Pass.
    pub fn forward_mut(&mut self, input: &[f32; IN]) -> [f32; OUT] {
        self.last_input = *input;
        self.forward(input)
    }

    /// Backward-Pass. `grad_output` ist `∂L/∂y`, das Ergebnis ist `∂L/∂x`.
    ///
    /// Die Gradienten der Parameter werden zu den bisherigen addiert:
    /// `∂L/∂W[j][i] = g[j] · x[i]`, `∂L/∂b[j] = g[j]`, `∂L/∂x[i] = Σ_j W[j][i] · g[j]`.
    pub fn backward_mut(&mut self, grad_output: &[f32; OUT]) -> [f32; IN] {
        for (j, &g) in grad_output.iter().enumerate() {
            self.grad_bias[j] += g;
            for (grad_w, &x) in self.grad_weights[j].iter_mut().zip(&self.last_input) {
                *grad_w += g * x;
            }
        }

        core::array::from_fn(|i| {
            self.weights
                .iter()
                .zip(grad_output)
                .map(|(row, &g)| row[i] * g)
                .sum()
        })
    }

    /// Setzt die aufsummierten Gradienten auf 0 zurück.
    pub fn zero_grad(&mut self) {
        self.grad_weights = [[0.0; IN]; OUT];
        self.grad_bias = [0.0; OUT];
    }

    pub const fn grad_weights(&self) -> &[[f32; IN]; OUT] {
        &self.grad_weights
    }

    pub const fn grad_bias(&self) -> &[f32; OUT] {
        &self.grad_bias
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
