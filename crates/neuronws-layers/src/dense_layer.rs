use crate::random::Rng;
use crate::scalar::Scalar;
use crate::traits::{
    BackwardData, BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer, Trainable,
};

pub struct DenseLayer<const IN: usize, const OUT: usize, T: Scalar = f32> {
    weights: [[T; IN]; OUT],
    bias: [T; OUT],
    /// Die Eingabe des letzten `forward_mut`, der Backward-Pass braucht sie für `∂L/∂W`.
    last_input: [T; IN],
    /// Die Ausgabe des letzten `forward_mut`, wird als Referenz zurückgegeben.
    output: [T; OUT],
    /// Aufsummierte Gradienten, bis `zero_grad` sie zurücksetzt.
    grad_weights: [[T; IN]; OUT],
    grad_bias: [T; OUT],
    /// Der Gradient für den Vorgänger, wird als Referenz zurückgegeben.
    grad_input: [T; IN],
}

impl<const IN: usize, const OUT: usize, T: Scalar> DenseLayer<IN, OUT, T> {
    pub const fn new(weights: [[T; IN]; OUT], bias: [T; OUT]) -> Self {
        Self {
            weights,
            bias,
            last_input: [T::ZERO; IN],
            output: [T::ZERO; OUT],
            grad_weights: [[T::ZERO; IN]; OUT],
            grad_bias: [T::ZERO; OUT],
            grad_input: [T::ZERO; IN],
        }
    }

    /// Gewichte gleichverteilt in `[-limit, limit]`, Bias 0.
    ///
    /// Zufällige Startwerte sind nötig, damit die Neuronen einer Schicht nicht alle dasselbe lernen.
    pub fn random(rng: &mut Rng, limit: T) -> Self {
        let weights = core::array::from_fn(|_| {
            core::array::from_fn(|_| limit * (T::TWO * T::from_f32(rng.next_f32()) - T::ONE))
        });
        Self::new(weights, [T::ZERO; OUT])
    }

    /// Wie `random`, mit der Xavier-Grenze `√(6 / (IN + OUT))`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn xavier(rng: &mut Rng) -> Self {
        Self::random(rng, (T::from_f32(6.0) / T::from_usize(IN + OUT)).sqrt())
    }

    pub const fn weights(&self) -> &[[T; IN]; OUT] {
        &self.weights
    }

    pub const fn bias(&self) -> &[T; OUT] {
        &self.bias
    }

    pub const fn grad_weights(&self) -> &[[T; IN]; OUT] {
        &self.grad_weights
    }

    pub const fn grad_bias(&self) -> &[T; OUT] {
        &self.grad_bias
    }
}

impl<const IN: usize, const OUT: usize, T: Scalar> Layer for DenseLayer<IN, OUT, T> {}

impl<const IN: usize, const OUT: usize, T: Scalar> ForwardData for DenseLayer<IN, OUT, T> {
    type ForwardInput = [T; IN];
    type ForwardOutput = [T; OUT];
}

impl<const IN: usize, const OUT: usize, T: Scalar> ForwardPath for DenseLayer<IN, OUT, T> {
    /// y[j] = Σ_i W[j][i] * x[i] + b[j], noch ohne Aktivierung.
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        core::array::from_fn(|j| {
            let dot = self.weights[j]
                .iter()
                .zip(input)
                .fold(T::ZERO, |sum, (&w, &x)| sum + w * x);
            dot + self.bias[j]
        })
    }
}

impl<const IN: usize, const OUT: usize, T: Scalar> ForwardPathMut for DenseLayer<IN, OUT, T> {
    /// Wie `forward`, merkt sich aber Ein- und Ausgabe.
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        self.last_input = *input;
        self.output = self.forward(input);
        &self.output
    }
}

impl<const IN: usize, const OUT: usize, T: Scalar> BackwardData for DenseLayer<IN, OUT, T> {
    /// Gradient bezüglich der Ausgabe, `∂L/∂y`.
    type BackwardInput = [T; OUT];
    /// Gradient bezüglich der Eingabe, `∂L/∂x`.
    type BackwardOutput = [T; IN];
}

impl<const IN: usize, const OUT: usize, T: Scalar> BackwardPathMut for DenseLayer<IN, OUT, T> {
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
                .fold(T::ZERO, |sum, (row, &g)| sum + row[i] * g)
        });
        &self.grad_input
    }
}

impl<const IN: usize, const OUT: usize, T: Scalar> Trainable for DenseLayer<IN, OUT, T> {
    type Scalar = T;

    fn visit_parameters(&mut self, visitor: &mut impl FnMut(&mut T, T)) {
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
        self.grad_weights = [[T::ZERO; IN]; OUT];
        self.grad_bias = [T::ZERO; OUT];
    }
}
