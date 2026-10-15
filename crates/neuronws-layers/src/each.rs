use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;

/// Wendet dieselbe parameterlose Operation `Op` auf jedes der `N` Teile eines Arrays an.
///
/// Aus `Op: [A] → [B]` wird `Each<Op, N>: [[A]; N] → [[B]; N]`. So bekommt man mit einem
/// vorhandenen Baustein eine Version für eine Achse mehr: Softmax über **jede Zeile** einer
/// Matrix, ohne dass Softmax etwas von Matrizen wissen muss.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))]
/// # {
/// use neuronws_layers::block::Block;
/// use neuronws_layers::each::Each;
/// use neuronws_layers::softmax_layer::Softmax;
/// use neuronws_layers::traits::ForwardPath;
///
/// // 2 Zeilen mit je 3 Werten: Softmax einzeln pro Zeile.
/// let softmax_per_row = Block::<Each<Softmax<3>, 2>>::new(Softmax::new());
/// let [first, second] = softmax_per_row.forward(&[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]);
/// assert!((first.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// assert!((second.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// # }
/// ```
///
/// Gedacht für Operationen ohne Parameter und ohne eigenen Zustand (Softmax, elementweise
/// Funktionen, Pooling). Eine Operation mit Zufall wie Dropout hat nur **eine** Maske und wäre
/// hier falsch. Operationen mit lernbaren Parametern (Dense) teilen sich die Gewichte über alle
/// Teile; dafür gibt es eigene Bausteine wie `LinearRows`.
#[derive(Debug, Clone, Copy)]
pub struct Each<Op, const N: usize> {
    op: Op,
}

impl<Op, const N: usize> Each<Op, N> {
    pub const fn new(op: Op) -> Self {
        Self { op }
    }

    pub const fn operation(&self) -> &Op {
        &self.op
    }
}

impl<Op: Operation<Params = ()>, const N: usize> Operation for Each<Op, N> {
    type Scalar = Op::Scalar;
    type Input = [Op::Input; N];
    type Output = [Op::Output; N];
    type Params = ();

    fn forward(&self, _params: &(), input: &Self::Input) -> Self::Output {
        core::array::from_fn(|i| self.op.forward(&(), &input[i]))
    }

    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        input: &Self::Input,
        output: &Self::Output,
        grad_output: &Self::Output,
    ) -> Self::Input {
        core::array::from_fn(|i| {
            self.op
                .backward(&(), &mut (), &input[i], &output[i], &grad_output[i])
        })
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

impl<Op: Operation<Params = ()>, const N: usize> Block<Each<Op, N>> {
    pub fn new(op: Op) -> Self {
        Self::from_parts(Each::new(op), ())
    }
}
