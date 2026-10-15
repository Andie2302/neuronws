use crate::block::{Block, Operation};
use crate::chain::Chain;
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;

/// Baut eine Schicht aus austauschbaren Teilen zusammen: **Berechnung**, **Startwerte** und
/// **Aktivierung**.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))]
/// # {
/// use neuronws_layers::activation::Tanh;
/// use neuronws_layers::activation_layer::Elementwise;
/// use neuronws_layers::dense_layer::Linear;
/// use neuronws_layers::init::Xavier;
/// use neuronws_layers::neurons::neurons;
/// use neuronws_layers::random::Rng;
/// use neuronws_layers::traits::ForwardPath;
///
/// let mut rng = Rng::new(1);
/// // 4 Eingänge, 3 Neuronen, Berechnung `Linear`, Startwerte `Xavier`, Aktivierung `Tanh`
/// let layer = neurons(Linear::<4, 3>::new(), &Xavier, Elementwise::new(Tanh), &mut rng);
/// let output = layer.forward(&[1.0, 0.0, -1.0, 0.5]);
/// assert!(output.iter().all(|y| y.abs() < 1.0));
/// # }
/// ```
///
/// Das Ergebnis ist eine [`Chain`] aus zwei Blöcken (erst die Berechnung, dann die Aktivierung).
/// Wie jede andere Schicht lässt es sich mit `.then(...)` weiter verketten und trainieren.
///
/// Als Aktivierung geht jede Operation, die `W` Werte auf `W` Werte abbildet und keine Parameter
/// hat: eine elementweise Funktion (`Elementwise::new(Tanh)`) oder auch `Softmax`.
pub fn neurons<Calc, Act, I, T, const IN: usize, const W: usize>(
    calculation: Calc,
    initializer: &I,
    activation: Act,
    rng: &mut Rng,
) -> Chain<Block<Calc>, Block<Act>>
where
    T: Scalar,
    I: Initializer,
    Calc: Operation<Scalar = T, Input = [T; IN], Output = [T; W]>,
    Act: Operation<Scalar = T, Input = [T; W], Output = [T; W]>,
{
    Chain::new(
        Block::initialized(calculation, initializer, rng),
        Block::initialized(activation, initializer, rng),
    )
}
