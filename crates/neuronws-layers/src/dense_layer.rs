use crate::random::Rng;
use crate::traits::{
    BackwardData, BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer, Trainable,
};

pub struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: [[f32; IN]; OUT],
    bias: [f32; OUT],
    /// Die Eingabe des letzten `forward_mut`, der Backward-Pass braucht sie für `∂L/∂W`.
    last_input: [f32; IN],
    /// Die Ausgabe des letzten `forward_mut`, wird als Referenz zurückgegeben.
    output: [f32; OUT],
    /// Aufsummierte Gradienten, bis `zero_grad` sie zurücksetzt.
    grad_weights: [[f32; IN]; OUT],
    grad_bias: [f32; OUT],
    /// Der Gradient für den Vorgänger, wird als Referenz zurückgegeben.
    grad_input: [f32; IN],
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub const fn new(weights: [[f32; IN]; OUT], bias: [f32; OUT]) -> Self {
        Self {
            weights,
            bias,
            last_input: [0.0; IN],
            output: [0.0; OUT],
            grad_weights: [[0.0; IN]; OUT],
            grad_bias: [0.0; OUT],
            grad_input: [0.0; IN],
        }
    }

    /// Gewichte gleichverteilt in `[-limit, limit]`, Bias 0.
    ///
    /// Zufällige Startwerte sind nötig, damit die Neuronen einer Schicht nicht alle dasselbe lernen.
    pub fn random(rng: &mut Rng, limit: f32) -> Self {
        let weights =
            core::array::from_fn(|_| core::array::from_fn(|_| rng.uniform(-limit, limit)));
        Self::new(weights, [0.0; OUT])
    }

    /// Wie `random`, mit der Xavier-Grenze `√(6 / (IN + OUT))`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn xavier(rng: &mut Rng) -> Self {
        Self::random(rng, crate::math::sqrt(6.0 / (IN + OUT) as f32))
    }

    pub const fn weights(&self) -> &[[f32; IN]; OUT] {
        &self.weights
    }

    pub const fn bias(&self) -> &[f32; OUT] {
        &self.bias
    }

    pub const fn grad_weights(&self) -> &[[f32; IN]; OUT] {
        &self.grad_weights
    }

    pub const fn grad_bias(&self) -> &[f32; OUT] {
        &self.grad_bias
    }
}

impl<const IN: usize, const OUT: usize> Layer for DenseLayer<IN, OUT> {}

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

impl<const IN: usize, const OUT: usize> ForwardPathMut for DenseLayer<IN, OUT> {
    /// Wie `forward`, merkt sich aber Ein- und Ausgabe.
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        self.last_input = *input;
        self.output = self.forward(input);
        &self.output
    }
}

impl<const IN: usize, const OUT: usize> BackwardData for DenseLayer<IN, OUT> {
    /// Gradient bezüglich der Ausgabe, `∂L/∂y`.
    type BackwardInput = [f32; OUT];
    /// Gradient bezüglich der Eingabe, `∂L/∂x`.
    type BackwardOutput = [f32; IN];
}

impl<const IN: usize, const OUT: usize> BackwardPathMut for DenseLayer<IN, OUT> {
    /// Addiert die Parameter-Gradienten zu den bisherigen und liefert `∂L/∂x`:
    /// `∂L/∂W[j][i] = g[j] · x[i]`, `∂L/∂b[j] = g[j]`, `∂L/∂x[i] = Σ_j W[j][i] · g[j]`.
    fn backward_mut(&mut self, grad_output: &Self::BackwardInput) -> &Self::BackwardOutput {
        for (j, &g) in grad_output.iter().enumerate() {
            self.grad_bias[j] += g;
            for (grad_w, &x) in self.grad_weights[j].iter_mut().zip(&self.last_input) {
                *grad_w += g * x;
            }
        }

        self.grad_input = core::array::from_fn(|i| {
            self.weights
                .iter()
                .zip(grad_output)
                .map(|(row, &g)| row[i] * g)
                .sum()
        });
        &self.grad_input
    }
}

impl<const IN: usize, const OUT: usize> Trainable for DenseLayer<IN, OUT> {
    fn visit_parameters(&mut self, visitor: &mut impl FnMut(&mut f32, f32)) {
        for (row, grad_row) in self.weights.iter_mut().zip(&self.grad_weights) {
            for (w, &g) in row.iter_mut().zip(grad_row) {
                visitor(w, g);
            }
        }
        for (b, &g) in self.bias.iter_mut().zip(&self.grad_bias) {
            visitor(b, g);
        }
    }

    fn zero_grad(&mut self) {
        self.grad_weights = [[0.0; IN]; OUT];
        self.grad_bias = [0.0; OUT];
    }
}
