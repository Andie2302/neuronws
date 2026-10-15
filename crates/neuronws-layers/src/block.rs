use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::traits::{
    BackwardData, BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer, Trainable,
};
use crate::values::Values;

/// Die reine Mathematik einer Schicht: Wie wird aus der Eingabe die Ausgabe, und wie fließt der
/// Gradient zurück?
///
/// Mehr muss man nicht schreiben, um eine neue Schicht zu definieren. Alles andere, die
/// Zwischenspeicher für den Backward-Pass, das Sammeln der Gradienten und alle Fähigkeiten
/// (`ForwardPath`, `BackwardPathMut`, `Trainable`, …), liefert [`Block`].
///
/// Eine Operation darf feste Einstellungen enthalten (zum Beispiel die Aktivierungsfunktion).
/// Was **gelernt** wird, steht in `Params`.
pub trait Operation {
    type Scalar: Scalar;
    type Input: Values<Self::Scalar>;
    type Output: Values<Self::Scalar>;
    /// Die lernbaren Parameter, `()` wenn es keine gibt.
    type Params: Values<Self::Scalar>;

    fn forward(&self, params: &Self::Params, input: &Self::Input) -> Self::Output;

    /// Liefert `∂L/∂Eingabe` und **addiert** `∂L/∂Parameter` zu `grads`.
    ///
    /// `output` ist das Ergebnis von `forward(params, input)`. Es wird mitgegeben, weil sich manche
    /// Ableitungen daraus billiger berechnen lassen (Sigmoid, Tanh, Softmax).
    fn backward(
        &self,
        params: &Self::Params,
        grads: &mut Self::Params,
        input: &Self::Input,
        output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input;

    /// Startwerte für die Parameter, gezogen mit dem gewählten `Initializer`.
    fn init_params<I: Initializer>(&self, initializer: &I, rng: &mut Rng) -> Self::Params;
}

/// Eine Operation mit allem, was eine Schicht sonst noch braucht: Parameter, ihre Gradienten und
/// die Zwischenspeicher für den Backward-Pass.
///
/// `Block<Operation>` ist für **jede** Operation ein `Layer` mit `ForwardPath`, `ForwardPathMut`,
/// `BackwardPathMut` und `Trainable`.
#[derive(Debug, Clone)]
pub struct Block<Op: Operation> {
    op: Op,
    params: Op::Params,
    grads: Op::Params,
    /// Die Eingabe des letzten `forward_mut`, der Backward-Pass braucht sie.
    last_input: Op::Input,
    /// Die Ausgabe des letzten `forward_mut`, wird als Referenz zurückgegeben.
    output: Op::Output,
    /// Der Gradient für den Vorgänger, wird als Referenz zurückgegeben.
    grad_input: Op::Input,
}

impl<Op: Operation> Block<Op> {
    /// Aus fertigen Parametern.
    pub fn from_parts(op: Op, params: Op::Params) -> Self {
        Self {
            op,
            params,
            grads: Op::Params::zeros(),
            last_input: Op::Input::zeros(),
            output: Op::Output::zeros(),
            grad_input: Op::Input::zeros(),
        }
    }

    /// Mit Startwerten, die der `Initializer` zieht.
    pub fn initialized<I: Initializer>(op: Op, initializer: &I, rng: &mut Rng) -> Self {
        let params = op.init_params(initializer, rng);
        Self::from_parts(op, params)
    }

    pub const fn operation(&self) -> &Op {
        &self.op
    }

    pub const fn params(&self) -> &Op::Params {
        &self.params
    }

    /// Die bisher aufsummierten Gradienten der Parameter.
    pub const fn grads(&self) -> &Op::Params {
        &self.grads
    }
}

impl<Op: Operation> Layer for Block<Op> {}

impl<Op: Operation> ForwardData for Block<Op> {
    type ForwardInput = Op::Input;
    type ForwardOutput = Op::Output;
}

impl<Op: Operation> ForwardPath for Block<Op> {
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        self.op.forward(&self.params, input)
    }
}

impl<Op: Operation> ForwardPathMut for Block<Op> {
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        self.last_input = *input;
        self.output = self.op.forward(&self.params, input);
        &self.output
    }
}

impl<Op: Operation> BackwardData for Block<Op> {
    type BackwardInput = Op::Output;
    type BackwardOutput = Op::Input;
}

impl<Op: Operation> BackwardPathMut for Block<Op> {
    fn backward_mut(&mut self, grad_output: &Self::BackwardInput) -> &Self::BackwardOutput {
        self.grad_input = self.op.backward(
            &self.params,
            &mut self.grads,
            &self.last_input,
            &self.output,
            grad_output,
        );
        &self.grad_input
    }
}

impl<Op: Operation> Trainable for Block<Op> {
    type Scalar = Op::Scalar;
    type Shape = Op::Params;

    const PARAMETER_COUNT: usize = <Op::Params as Values<Op::Scalar>>::LEN;

    fn visit_parameters(&mut self, visitor: &mut impl FnMut(&mut Op::Scalar, Op::Scalar)) {
        for index in 0..Self::PARAMETER_COUNT {
            visitor(self.params.value_mut(index), self.grads.value(index));
        }
    }

    fn zero_grad(&mut self) {
        self.grads = Op::Params::zeros();
    }
}
