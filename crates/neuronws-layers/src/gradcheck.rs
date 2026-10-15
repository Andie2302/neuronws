//! Prüft Ableitungen gegen numerische Ableitungen: von einer [`Operation`], einer [`Activation`]
//! und einer [`Loss`]-Funktion.
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

use crate::activation::Activation;
use crate::block::Operation;
use crate::loss::Loss;
use crate::scalar::Scalar;
use crate::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};
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

/// Prüft `derivative` und `derivative_with_output` einer [`Activation`] an den gegebenen Stellen
/// gegen die numerische Ableitung `(φ(z + h) − φ(z − h)) / 2h` (gerechnet mit `f64`).
///
/// Stellen mit Knick (bei ReLU die 0) sollten nicht dabei sein, dort ist die Ableitung nicht
/// definiert.
///
/// # Panics
/// Wenn eine Ableitung um mehr als `tolerance · (1 + |Wert|)` abweicht, mit der Angabe, wo.
pub fn assert_activation_matches<A: Activation>(activation: &A, points: &[f64], tolerance: f64) {
    let h = 1e-6;
    for &z in points {
        let phi = |x: f64| activation.apply(x);
        let numeric = (phi(z + h) - phi(z - h)) / (2.0 * h);
        let allowed = tolerance * (1.0 + numeric.abs());
        let analytic = activation.derivative(z);
        assert!(
            (analytic - numeric).abs() <= allowed,
            "φ'({z}): analytisch {analytic}, numerisch {numeric}"
        );
        let with_output = activation.derivative_with_output(z, phi(z));
        assert!(
            (with_output - numeric).abs() <= allowed,
            "φ'({z}) mit bekanntem Ergebnis: {with_output}, numerisch {numeric}"
        );
    }
}

/// Prüft `Loss::gradient` gegen die numerische Ableitung von `Loss::value` nach jeder Ausgabe.
///
/// # Panics
/// Wenn ein Gradient um mehr als `tolerance · (1 + |Wert|)` abweicht, mit der Angabe, welcher.
pub fn assert_loss_gradient_matches<T: Scalar, V: Values<T>, L: Loss>(
    loss: &L,
    output: V,
    target: V,
    epsilon: T,
    tolerance: f64,
) {
    let gradient = loss.gradient(&output, &target);
    let two_eps = T::TWO * epsilon;
    for index in 0..V::LEN {
        let (mut up, mut down) = (output, output);
        *up.value_mut(index) += epsilon;
        *down.value_mut(index) -= epsilon;
        let numeric = (loss.value(&up, &target) - loss.value(&down, &target)) / two_eps;
        let (a, n) = (gradient.value(index).to_f64(), numeric.to_f64());
        let allowed = tolerance * (1.0 + a.abs().max(n.abs()));
        assert!(
            (a - n).abs() <= allowed,
            "Ableitung nach Ausgabe[{index}]: analytisch {a}, numerisch {n}"
        );
    }
}

/// Prüft die Ableitungen eines ganzen **Netzes** (einer `Chain`, eines `Residual`, eines `Fork`, …)
/// gegen numerische Ableitungen, nach der Eingabe und nach jedem Parameter.
///
/// Es arbeitet wie [`assert_gradients_match`] mit dem Verlust `L = Σ c[k] · Ausgabe[k]`. Gerechnet
/// wird einmal über den Lernweg (`forward_mut`, `backward_mut`) und die numerischen Ableitungen
/// über den Einsatzweg (`forward`). Beide müssen also dieselbe Funktion berechnen: Das gilt nicht
/// für Schichten mit Zufall (Dropout), bei denen sie sich absichtlich unterscheiden.
///
/// Danach sind die Gradienten des Modells wieder auf 0 gesetzt.
///
/// # Panics
/// Wenn eine Ableitung nicht stimmt, mit der Angabe, welche.
pub fn assert_layer_gradients_match<M, T, I, O>(model: &mut M, input: I, epsilon: T, tolerance: f64)
where
    T: Scalar,
    I: Values<T>,
    O: Values<T>,
    M: ForwardPath<ForwardInput = I, ForwardOutput = O>
        + ForwardPathMut<ForwardInput = I, ForwardOutput = O>
        + BackwardPathMut<BackwardInput = O, BackwardOutput = I>
        + Trainable<Scalar = T>,
{
    let weight = |k: usize| T::from_f32(((k * 7 + 3) % 11) as f32 / 3.0 - 1.5);
    let loss = |model: &M, input: &I| {
        let output = model.forward(input);
        (0..O::LEN).fold(T::ZERO, |sum, k| sum + weight(k) * output.value(k))
    };

    model.zero_grad();
    model.forward_mut(&input);
    let mut grad_output = O::zeros();
    for k in 0..O::LEN {
        *grad_output.value_mut(k) = weight(k);
    }
    let grad_input = *model.backward_mut(&grad_output);
    let analytic = gradients_of(model);
    let two_eps = T::TWO * epsilon;

    let check = |what: &str, index: usize, a: T, n: T| {
        let (a, n) = (a.to_f64(), n.to_f64());
        let allowed = tolerance * (1.0 + a.abs().max(n.abs()));
        assert!(
            (a - n).abs() <= allowed,
            "Ableitung nach {what}[{index}]: analytisch {a}, numerisch {n}"
        );
    };

    for index in 0..I::LEN {
        let (mut up, mut down) = (input, input);
        *up.value_mut(index) += epsilon;
        *down.value_mut(index) -= epsilon;
        let numeric = (loss(model, &up) - loss(model, &down)) / two_eps;
        check("Eingabe", index, grad_input.value(index), numeric);
    }

    for index in 0..M::PARAMETER_COUNT {
        let mut shifted_loss = |delta: T| {
            let mut position = 0;
            model.visit_parameters(&mut |parameter, _| {
                if position == index {
                    *parameter += delta;
                }
                position += 1;
            });
            let value = loss(model, &input);
            let mut position = 0;
            model.visit_parameters(&mut |parameter, _| {
                if position == index {
                    *parameter -= delta;
                }
                position += 1;
            });
            value
        };
        let numeric = (shifted_loss(epsilon) - shifted_loss(-epsilon)) / two_eps;
        check("Parameter", index, analytic.value(index), numeric);
    }
    model.zero_grad();
}

/// Die aufsummierten Gradienten des Modells als Gruppe von der Form `Shape`.
fn gradients_of<M: Trainable>(model: &M) -> M::Shape {
    let mut gradients = <M::Shape as Values<M::Scalar>>::zeros();
    let mut index = 0;
    model.inspect_parameters(&mut |_, gradient| {
        *gradients.value_mut(index) = gradient;
        index += 1;
    });
    gradients
}
