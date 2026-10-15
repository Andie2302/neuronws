//! Prüft die Ableitung einer [`Operation`] gegen numerische Ableitungen.
//!
//! Wer eine neue Operation schreibt, kann damit beweisen, dass sein `backward` stimmt:
//!
//! ```
//! # use neuronws_layers::dense_layer::Linear;
//! # use neuronws_layers::values::Values;
//! use neuronws_layers::gradcheck::assert_gradients_match;
//!
//! let params = ([[0.5_f32, -1.0], [2.0, 0.3]], [0.1, -0.2]);
//! assert_gradients_match(&Linear::<2, 2>::new(), params, [1.5, -0.5], 1e-2, 1e-3);
//! ```

use crate::block::Operation;
use crate::scalar::Scalar;
use crate::values::Values;

/// Prüft `∂L/∂Eingabe` und `∂L/∂Parameter` für den Verlust `L = Σ c[k] · Ausgabe[k]` mit
/// festen, verschieden großen Gewichten `c`.
///
/// Die numerische Ableitung ist `(L(p + ε) − L(p − ε)) / 2ε`. Ein Wert gilt als richtig, wenn der
/// Unterschied höchstens `tolerance · (1 + |Wert|)` beträgt.
///
/// # Panics
/// Wenn eine Ableitung nicht stimmt, mit der Angabe, welche.
pub fn assert_gradients_match<Op: Operation>(
    op: &Op,
    params: Op::Params,
    input: Op::Input,
    epsilon: Op::Scalar,
    tolerance: f64,
) {
    let weight = |k: usize| Op::Scalar::from_f32(((k * 7 + 3) % 11) as f32 / 3.0 - 1.5);
    let output_len = <Op::Output as Values<Op::Scalar>>::LEN;
    let loss = |params: &Op::Params, input: &Op::Input| {
        let output = op.forward(params, input);
        (0..output_len).fold(Op::Scalar::ZERO, |sum, k| sum + weight(k) * output.value(k))
    };

    let output = op.forward(&params, &input);
    let mut grad_output = Op::Output::zeros();
    for k in 0..output_len {
        *grad_output.value_mut(k) = weight(k);
    }
    let mut grads = Op::Params::zeros();
    let grad_input = op.backward(&params, &mut grads, &input, &output, &grad_output);

    let two_eps = Op::Scalar::TWO * epsilon;
    let check = |what: &str, index: usize, analytic: Op::Scalar, numeric: Op::Scalar| {
        let (a, n) = (analytic.to_f64(), numeric.to_f64());
        let allowed = tolerance * (1.0 + a.abs().max(n.abs()));
        assert!(
            (a - n).abs() <= allowed,
            "Ableitung nach {what}[{index}]: analytisch {a}, numerisch {n}"
        );
    };

    for index in 0..<Op::Input as Values<Op::Scalar>>::LEN {
        let (mut up, mut down) = (input, input);
        *up.value_mut(index) += epsilon;
        *down.value_mut(index) -= epsilon;
        let numeric = (loss(&params, &up) - loss(&params, &down)) / two_eps;
        check("Eingabe", index, grad_input.value(index), numeric);
    }
    for index in 0..<Op::Params as Values<Op::Scalar>>::LEN {
        let (mut up, mut down) = (params, params);
        *up.value_mut(index) += epsilon;
        *down.value_mut(index) -= epsilon;
        let numeric = (loss(&up, &input) - loss(&down, &input)) / two_eps;
        check("Parameter", index, grads.value(index), numeric);
    }
}
