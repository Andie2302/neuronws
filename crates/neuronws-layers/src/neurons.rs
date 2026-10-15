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
/// Als Aktivierung geht jede parameterlose Operation, die die Form der Berechnung übernimmt: eine
/// elementweise Funktion (`Elementwise::new(Tanh)`) oder auch `Softmax`.
///
/// # Auch für Bilder
///
/// Der Helfer kennt keine Formen, deshalb funktioniert er für **jede**: Bei einer Faltung ist die
/// Berechnung eine `Conv2d` und die Aktivierung eine `Elementwise` über das Bild. Welche Form die
/// Aktivierung hat, leitet der Compiler aus der Berechnung ab, man muss sie nicht hinschreiben.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))]
/// # {
/// use neuronws_layers::activation::Relu;
/// use neuronws_layers::activation_layer::Elementwise;
/// use neuronws_layers::conv::Conv2dValid;
/// use neuronws_layers::init::He;
/// use neuronws_layers::neurons::neurons;
/// use neuronws_layers::random::Rng;
/// use neuronws_layers::traits::ForwardPath;
///
/// let mut rng = Rng::new(1);
/// // Faltung (1 Kanal → 2 Kanäle, 3×3-Kern auf 5×5-Bild) und ReLU in einem Aufruf
/// let layer = neurons(
///     Conv2dValid::<1, 2, 5, 5, 3, 3, 3, 3>::new(),
///     &He,
///     Elementwise::new(Relu),
///     &mut rng,
/// );
/// let output = layer.forward(&[[[1.0; 5]; 5]]);
/// assert_eq!((output.len(), output[0].len(), output[0][0].len()), (2, 3, 3));
/// # }
/// ```
pub fn neurons<Calc, Act, I, T>(
    calculation: Calc,
    initializer: &I,
    activation: Act,
    rng: &mut Rng,
) -> Chain<Block<Calc>, Block<Act>>
where
    T: Scalar,
    I: Initializer,
    Calc: Operation<Scalar = T>,
    Act: Operation<Scalar = T, Input = Calc::Output>,
{
    Chain::new(
        Block::initialized(calculation, initializer, rng),
        Block::initialized(activation, initializer, rng),
    )
}
