//! Faltung und Pooling auf Bildern.

use neuronws_layers::conv::{Conv2dValid, Conv2dValidLayer};
use neuronws_layers::gradcheck::assert_gradients_match;
use neuronws_layers::init::Constant;
use neuronws_layers::pooling::{AvgPool2d, AvgPool2dLayer, MaxPool2d, MaxPool2dLayer};
use neuronws_layers::random::Rng;
use neuronws_layers::scalar::Scalar;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};
use neuronws_layers::values::Values;

/// Füllt eine beliebige Form mit Zufallszahlen in (−1, 1).
fn random<V: Values<f64>>(rng: &mut Rng) -> V {
    let mut v = V::zeros();
    for i in 0..V::LEN {
        *v.value_mut(i) = f64::from(rng.uniform(-1.0, 1.0));
    }
    v
}

fn to_f32<V: Values<f64>, W: Values<f32>>(v: &V) -> W {
    let mut w = W::zeros();
    for i in 0..V::LEN {
        *w.value_mut(i) = v.value(i) as f32;
    }
    w
}

// ---------------------------------------------------------------------------------------------
// Faltung
// ---------------------------------------------------------------------------------------------

#[test]
fn a_difference_kernel_gives_the_known_result() {
    // y = x[r][c] − x[r+1][c+1]; im Bild 1..9 ist das überall −4.
    let kernel = [[[[1.0, 0.0], [0.0, -1.0]]]];
    let conv = Conv2dValidLayer::<1, 1, 3, 3, 2, 2, 2, 2>::new(kernel, [0.0]);
    let image = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]];
    assert_eq!(conv.forward(&image), [[[-4.0, -4.0], [-4.0, -4.0]]]);
}

#[test]
fn bias_is_added_everywhere_and_channels_are_summed() {
    // Zwei Eingabekanäle, ein 1×1-Kern pro Kanal: y = 2·x0 + 3·x1 + 0.5
    let kernels = [[[[2.0]], [[3.0]]]];
    let conv = Conv2dValidLayer::<2, 1, 2, 2, 1, 1, 2, 2>::new(kernels, [0.5]);
    let image = [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]];
    assert_eq!(
        conv.forward(&image),
        [[[32.5, 64.5], [96.5, 128.5]]] // 2·1+3·10+0.5, 2·2+3·20+0.5, ...
    );
}

#[test]
fn every_output_channel_has_its_own_kernel() {
    let kernels = [[[[1.0]]], [[[-1.0]]]];
    let conv = Conv2dValidLayer::<1, 2, 1, 2, 1, 1, 1, 2>::new(kernels, [0.0, 10.0]);
    assert_eq!(conv.forward(&[[[3.0, 4.0]]]), [[[3.0, 4.0]], [[7.0, 6.0]]]);
}

#[test]
fn the_same_kernel_finds_a_pattern_anywhere() {
    // Ein Kern, der "senkrechte Kante hell|dunkel" sucht: links +1, rechts −1.
    let kernel = [[[[1.0, -1.0], [1.0, -1.0]]]];
    let conv = Conv2dValidLayer::<1, 1, 3, 4, 2, 2, 2, 3>::new(kernel, [0.0]);
    let image = [[
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]];
    let [out] = conv.forward(&image);
    // Die Kante liegt zwischen Spalte 1 und 2: stärkste Antwort an Ausgabespalte 1.
    assert_eq!(out[0], [-2.0, 2.0, 0.0]);
    assert_eq!(out[1], [-2.0, 2.0, 0.0]);
}

#[test]
fn kernel_gradient_sums_the_contribution_of_every_position() {
    let mut conv = Conv2dValidLayer::<1, 1, 3, 3, 2, 2, 2, 2>::new([[[[0.0; 2]; 2]]], [0.0]);
    let image = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]];
    conv.forward_mut(&image);
    let grad_input = *conv.backward_mut(&[[[1.0, 1.0], [1.0, 1.0]]]);

    // ∂L/∂K[p][q] = Σ über alle 4 Stellen von x[r+p][c+q]
    assert_eq!(conv.grad_kernels(), &[[[[12.0, 16.0], [24.0, 28.0]]]]);
    assert_eq!(conv.grad_bias(), &[4.0]);
    // Kern ist 0, also fließt nichts zur Eingabe zurück.
    assert_eq!(grad_input, [[[0.0; 3]; 3]]);
}

#[test]
fn input_gradient_is_the_kernel_spread_over_the_windows() {
    let kernel = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let mut conv = Conv2dValidLayer::<1, 1, 3, 3, 2, 2, 2, 2>::new(kernel, [0.0]);
    conv.forward_mut(&[[[0.0; 3]; 3]]);
    let grad_input = *conv.backward_mut(&[[[1.0, 1.0], [1.0, 1.0]]]);
    // Die Mitte (1,1) liegt in allen vier Fenstern: 4+3+2+1 = 10.
    assert_eq!(
        grad_input,
        [[[1.0, 3.0, 2.0], [4.0, 10.0, 6.0], [3.0, 7.0, 4.0]]]
    );
}

#[test]
fn conv_gradients_match_numeric_ones_f64() {
    let mut rng = Rng::new(3);
    // 2 Eingabekanäle, 3 Ausgabekanäle, 5×4-Bild, 3×2-Kern → 3×3 heraus
    let params = (
        random::<[[[[f64; 2]; 3]; 2]; 3]>(&mut rng),
        random::<[f64; 3]>(&mut rng),
    );
    let input = random::<[[[f64; 4]; 5]; 2]>(&mut rng);
    assert_gradients_match(
        &Conv2dValid::<2, 3, 5, 4, 3, 2, 3, 3, f64>::new(),
        params,
        input,
        1e-6,
        1e-7,
    );
}

#[test]
fn conv_gradients_match_numeric_ones_f32() {
    let mut rng = Rng::new(4);
    let params64 = (
        random::<[[[[f64; 2]; 2]; 1]; 2]>(&mut rng),
        random::<[f64; 2]>(&mut rng),
    );
    let input64 = random::<[[[f64; 3]; 3]; 1]>(&mut rng);
    assert_gradients_match(
        &Conv2dValid::<1, 2, 3, 3, 2, 2, 2, 2>::new(),
        (to_f32(&params64.0), to_f32(&params64.1)),
        to_f32::<_, [[[f32; 3]; 3]; 1]>(&input64),
        1e-2,
        2e-3,
    );
}

#[test]
fn initialization_fills_kernels_and_leaves_the_bias_at_zero() {
    let block = neuronws_layers::block::Block::initialized(
        Conv2dValid::<2, 3, 4, 4, 2, 2, 3, 3>::new(),
        &Constant(0.25),
        &mut Rng::new(1),
    );
    assert_eq!(block.kernels(), &[[[[0.25; 2]; 2]; 2]; 3]);
    assert_eq!(block.bias(), &[0.0; 3]);
    assert_eq!(
        <Conv2dValidLayer<2, 3, 4, 4, 2, 2, 3, 3> as Trainable>::PARAMETER_COUNT,
        3 * 2 * 2 * 2 + 3
    );
}

// ---------------------------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------------------------

#[test]
fn max_pooling_keeps_the_largest_value_of_each_window() {
    let pool = MaxPool2dLayer::<1, 4, 4, 2, 2, 2>::new();
    let image = [[
        [1.0, 2.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 5.0],
        [0.0, 0.0, 6.0, 0.0],
        [7.0, 0.0, 0.0, 0.0],
    ]];
    assert_eq!(pool.forward(&image), [[[4.0, 5.0], [7.0, 6.0]]]);
}

#[test]
fn pooling_drops_an_incomplete_border() {
    let pool = MaxPool2dLayer::<1, 5, 5, 2, 2, 2>::new();
    let mut image = [[[0.0; 5]; 5]];
    image[0][4][4] = 99.0; // in der abgeschnittenen Zeile/Spalte
    image[0][1][1] = 1.0;
    assert_eq!(pool.forward(&image), [[[1.0, 0.0], [0.0, 0.0]]]);
}

#[test]
fn max_pooling_sends_the_gradient_only_to_the_maximum() {
    let mut pool = MaxPool2dLayer::<1, 4, 4, 2, 2, 2>::new();
    let image = [[
        [1.0, 2.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 5.0],
        [0.0, 0.0, 6.0, 0.0],
        [7.0, 0.0, 0.0, 0.0],
    ]];
    pool.forward_mut(&image);
    let grad = *pool.backward_mut(&[[[1.0, 2.0], [3.0, 4.0]]]);
    assert_eq!(
        grad,
        [[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 4.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ]]
    );
}

#[test]
fn on_ties_the_first_value_gets_the_gradient() {
    let mut pool = MaxPool2dLayer::<1, 2, 2, 2, 1, 1>::new();
    pool.forward_mut(&[[[5.0, 5.0], [5.0, 5.0]]]);
    let grad = *pool.backward_mut(&[[[1.0]]]);
    assert_eq!(grad, [[[1.0, 0.0], [0.0, 0.0]]]);
}

#[test]
fn average_pooling_averages_and_spreads_the_gradient_evenly() {
    let mut pool = AvgPool2dLayer::<1, 2, 4, 2, 1, 2>::new();
    let out = *pool.forward_mut(&[[[1.0, 3.0, 0.0, 0.0], [5.0, 7.0, 4.0, 8.0]]]);
    assert_eq!(out, [[[4.0, 3.0]]]);
    let grad = *pool.backward_mut(&[[[4.0, 8.0]]]);
    assert_eq!(grad, [[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]]);
}

#[test]
fn pooling_works_on_every_channel_separately() {
    let pool = MaxPool2dLayer::<2, 2, 2, 2, 1, 1>::new();
    assert_eq!(
        pool.forward(&[[[1.0, 2.0], [3.0, 4.0]], [[9.0, 0.0], [0.0, 0.0]]]),
        [[[4.0]], [[9.0]]]
    );
}

#[test]
fn pooling_gradients_match_numeric_ones() {
    // Eindeutig verschiedene, weit auseinanderliegende Werte, damit das Maximum stabil bleibt.
    let mut image = [[[0.0_f64; 6]; 4]; 2];
    for i in 0..<[[[f64; 6]; 4]; 2] as Values<f64>>::LEN {
        *Values::<f64>::value_mut(&mut image, i) = (i * 7 % 48) as f64 * 0.5;
    }
    assert_gradients_match(
        &MaxPool2d::<2, 4, 6, 2, 2, 3, f64>::new(),
        (),
        image,
        1e-3,
        1e-6,
    );
    assert_gradients_match(
        &AvgPool2d::<2, 4, 6, 2, 2, 3, f64>::new(),
        (),
        image,
        1e-3,
        1e-6,
    );
}

#[test]
fn scalar_type_is_generic() {
    fn forward<T: Scalar>() -> T {
        let pool = AvgPool2dLayer::<1, 2, 2, 2, 1, 1, T>::new();
        pool.forward(&[[[T::ONE, T::TWO], [T::ONE, T::TWO]]])[0][0][0]
    }
    assert_eq!(forward::<f32>(), 1.5);
    assert_eq!(forward::<f64>(), 1.5);
}
