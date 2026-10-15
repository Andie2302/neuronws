//! Dieselben Prüfungen laufen für `f32` und `f64`: Rückwärtsrechnung, Verlust, Optimierer.
//!
//! Jede Prüfung ist eine generische Funktion. Die Tests unten rufen sie mit beiden Typen auf,
//! mit je passender Schrittweite und Toleranz.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::Tanh;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::{Chain, LayerExt};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{BinaryCrossEntropy, CategoricalCrossEntropy, Loss, MeanSquaredError};
use neuronws_layers::optimizer::{Momentum, Optimizer, Sgd};
use neuronws_layers::random::Rng;
use neuronws_layers::scalar::Scalar;
use neuronws_layers::softmax_layer::SoftmaxLayer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

/// Schrittweite und Toleranz der numerischen Ableitung, passend zur Genauigkeit des Typs.
struct Precision {
    eps: f32,
    tolerance: f64,
}

const F32: Precision = Precision {
    eps: 1e-2,
    tolerance: 2e-3,
};
const F64: Precision = Precision {
    eps: 1e-5,
    tolerance: 1e-7,
};

fn t<T: Scalar>(x: f32) -> T {
    T::from_f32(x)
}

fn assert_close<T: Scalar + Into<f64>>(analytic: T, numeric: T, tolerance: f64, what: &str) {
    let (a, n): (f64, f64) = (analytic.into(), numeric.into());
    assert!(
        (a - n).abs() < tolerance,
        "{what}: analytisch {a}, numerisch {n}"
    );
}

/// Mittlere Ableitung von `f` an `x`: `(f(x + eps) − f(x − eps)) / 2 eps`.
fn numeric<T: Scalar>(eps: T, f: impl Fn(T) -> T) -> T {
    (f(eps) - f(-eps)) / (T::TWO * eps)
}

fn check_dense_gradients<T: Scalar + Into<f64>>(p: &Precision) {
    let w: [[T; 2]; 3] = [[0.5, -1.0], [2.0, 0.3], [-0.7, 1.1]].map(|row| row.map(t::<T>));
    let b: [T; 3] = [0.1, -0.2, 0.3].map(t::<T>);
    let x: [T; 2] = [1.5, -0.5].map(t::<T>);
    let target: [T; 3] = [1.0, 0.0, -1.0].map(t::<T>);
    let eps = t::<T>(p.eps);

    let loss = |w: [[T; 2]; 3], b: [T; 3], x: [T; 2]| {
        MeanSquaredError.value(&DenseLayer::new(w, b).forward(&x), &target)
    };

    let mut layer = DenseLayer::<2, 3, T>::new(w, b);
    let y = *layer.forward_mut(&x);
    let grad_input = *layer.backward_mut(&MeanSquaredError.gradient(&y, &target));

    for j in 0..3 {
        for i in 0..2 {
            let n = numeric(eps, |d| {
                let mut w = w;
                w[j][i] += d;
                loss(w, b, x)
            });
            assert_close(layer.grad_weights()[j][i], n, p.tolerance, "grad_weights");
        }
        let n = numeric(eps, |d| {
            let mut b = b;
            b[j] += d;
            loss(w, b, x)
        });
        assert_close(layer.grad_bias()[j], n, p.tolerance, "grad_bias");
    }
    for i in 0..2 {
        let n = numeric(eps, |d| {
            let mut x = x;
            x[i] += d;
            loss(w, b, x)
        });
        assert_close(grad_input[i], n, p.tolerance, "grad_input");
    }
}

type Classifier<T> = Chain<
    Chain<Chain<DenseLayer<2, 4, T>, ActivationLayer<4, Tanh, T>>, DenseLayer<4, 3, T>>,
    SoftmaxLayer<3, T>,
>;

fn classifier<T: Scalar>() -> Classifier<T> {
    let mut rng = Rng::new(11);
    DenseLayer::<2, 4, T>::xavier(&mut rng)
        .then(ActivationLayer::<4, _, T>::new(Tanh))
        .then(DenseLayer::<4, 3, T>::xavier(&mut rng))
        .then(SoftmaxLayer::<3, T>::new())
}

/// Der ganze Weg rückwärts durch Dense → Tanh → Dense → Softmax, mit Kreuzentropie am Ende.
fn check_full_network_input_gradient<T: Scalar + Into<f64>>(p: &Precision) {
    let x: [T; 2] = [0.7, -0.4].map(t::<T>);
    let target: [T; 3] = [0.0, 1.0, 0.0].map(t::<T>);
    let eps = t::<T>(p.eps);

    let mut net = classifier::<T>();
    let y = *net.forward_mut(&x);
    let analytic = *net.backward_mut(&CategoricalCrossEntropy.gradient(&y, &target));

    let reference = classifier::<T>();
    for (i, a) in analytic.iter().enumerate() {
        let n = numeric(eps, |d| {
            let mut x = x;
            x[i] += d;
            CategoricalCrossEntropy.value(&reference.forward(&x), &target)
        });
        assert_close(*a, n, p.tolerance, "Gradient durch das ganze Netz");
    }
}

fn check_binary_cross_entropy_gradient<T: Scalar + Into<f64>>(p: &Precision) {
    let y: [T; 3] = [0.3, 0.8, 0.55].map(t::<T>);
    let target: [T; 3] = [1.0, 1.0, 0.0].map(t::<T>);
    let eps = t::<T>(p.eps / 10.0);
    let analytic = BinaryCrossEntropy.gradient(&y, &target);
    for (i, a) in analytic.iter().enumerate() {
        let n = numeric(eps, |d| {
            let mut y = y;
            y[i] += d;
            BinaryCrossEntropy.value(&y, &target)
        });
        assert_close(*a, n, p.tolerance * 10.0, "BCE");
    }
}

/// `v ← μ v − η g`, `p ← p + v`, zweimal hintereinander mit demselben Gradienten.
fn check_momentum_formula<T: Scalar + Into<f64>>(p: &Precision) {
    let (lr, mu) = (t::<T>(0.1), t::<T>(0.5));
    let mut layer = DenseLayer::<1, 1, T>::new([[T::ONE]], [T::ZERO]);
    let mut momentum = Momentum::<2, T>::new(lr, mu);
    let (mut w, mut b) = (T::ONE, T::ZERO);
    let (mut vw, mut vb) = (T::ZERO, T::ZERO);

    for _ in 0..3 {
        layer.forward_mut(&[T::TWO]);
        layer.backward_mut(&[T::ONE]); // grad_w = 2, grad_b = 1
        momentum.step(&mut layer, 1);

        vw = mu * vw - lr * T::TWO;
        vb = mu * vb - lr * T::ONE;
        w += vw;
        b += vb;
        assert_close(layer.weights()[0][0], w, p.tolerance, "Gewicht");
        assert_close(layer.bias()[0], b, p.tolerance, "Bias");
    }
}

fn check_optimizers_learn_a_line<T: Scalar + Into<f64>>(p: &Precision) {
    fn train<T: Scalar, O: Optimizer<T>>(mut optimizer: O) -> DenseLayer<1, 1, T> {
        let mut layer = DenseLayer::<1, 1, T>::new([[T::ZERO]], [T::ZERO]);
        let data = [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x| ([t::<T>(x)], [t::<T>(2.0 * x + 1.0)]));
        for _ in 0..400 {
            for (x, target) in &data {
                let y = *layer.forward_mut(x);
                layer.backward_mut(&MeanSquaredError.gradient(&y, target));
            }
            optimizer.step(&mut layer, data.len());
        }
        layer
    }

    for layer in [
        train::<T, _>(Sgd::new(t(0.05))),
        train::<T, _>(Momentum::<2, T>::new(t(0.02), t(0.9))),
    ] {
        assert_close(
            layer.weights()[0][0],
            t::<T>(2.0),
            p.tolerance.max(1e-3),
            "w",
        );
        assert_close(layer.bias()[0], t::<T>(1.0), p.tolerance.max(1e-3), "b");
    }
}

macro_rules! for_both_scalars {
    ($($check:ident => $f32_name:ident, $f64_name:ident;)*) => {$(
        #[test] fn $f32_name() { $check::<f32>(&F32); }
        #[test] fn $f64_name() { $check::<f64>(&F64); }
    )*};
}

for_both_scalars! {
    check_dense_gradients => dense_gradients_f32, dense_gradients_f64;
    check_full_network_input_gradient => full_network_gradient_f32, full_network_gradient_f64;
    check_binary_cross_entropy_gradient => bce_gradient_f32, bce_gradient_f64;
    check_momentum_formula => momentum_formula_f32, momentum_formula_f64;
    check_optimizers_learn_a_line => optimizers_learn_a_line_f32, optimizers_learn_a_line_f64;
}
