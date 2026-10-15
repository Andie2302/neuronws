//! Netze mit Matrizen und Bildern als Ein- und Ausgabe, nicht nur flachen Vektoren.

use neuronws_layers::activation::{Identity, Relu};
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::gradcheck::assert_gradients_match;
use neuronws_layers::loss::{Loss, MeanSquaredError};
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::reshape::{Reshape, ReshapeLayer};
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut};

type Image = [[[f32; 3]; 2]; 2]; // 2 Kanäle, 2 Zeilen, 3 Spalten

fn image() -> Image {
    [
        [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
        [[0.5, -0.25, 2.0], [-1.5, 1.0, 0.75]],
    ]
}

#[test]
fn reshape_keeps_the_order_in_both_directions() {
    let flatten = ReshapeLayer::<Image, [f32; 12]>::new();
    let flat = flatten.forward(&image());
    assert_eq!(
        flat,
        [
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.5, -0.25, 2.0, -1.5, 1.0, 0.75
        ]
    );

    let fold = ReshapeLayer::<[f32; 12], Image>::new();
    assert_eq!(fold.forward(&flat), image());
}

#[test]
fn reshape_gradients_pass_through_unchanged() {
    assert_gradients_match(&Reshape::<Image, [f32; 12]>::new(), (), image(), 1e-2, 2e-3);
    assert_gradients_match(
        &Reshape::<[f32; 4], [[f32; 2]; 2]>::new(),
        (),
        [1.0, 2.0, 3.0, 4.0],
        1e-2,
        2e-3,
    );
}

#[test]
fn activations_work_on_any_shape() {
    let relu = Block::<Elementwise<Image, Relu>>::new(Relu);
    assert_eq!(
        relu.forward(&image()),
        [
            [[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]],
            [[0.5, 0.0, 2.0], [0.0, 1.0, 0.75]],
        ]
    );
    assert_gradients_match(&Elementwise::<Image, _>::new(Relu), (), image(), 1e-2, 2e-3);
}

#[test]
fn a_loss_sees_matrices_as_flat_lists() {
    let y = [[1.0_f32, 2.0], [3.0, 4.0]];
    let t = [[1.0_f32, 0.0], [5.0, 4.0]];
    let (flat_y, flat_t) = ([1.0_f32, 2.0, 3.0, 4.0], [1.0_f32, 0.0, 5.0, 4.0]);

    assert_eq!(
        MeanSquaredError.value(&y, &t),
        MeanSquaredError.value(&flat_y, &flat_t)
    );
    let g = MeanSquaredError.gradient(&y, &t);
    let flat_g = MeanSquaredError.gradient(&flat_y, &flat_t);
    assert_eq!(g, [[flat_g[0], flat_g[1]], [flat_g[2], flat_g[3]]]);
}

#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn cross_entropy_losses_accept_matrices_too() {
    use neuronws_layers::loss::{BinaryCrossEntropy, CategoricalCrossEntropy};

    let y = [[0.9_f32, 0.2], [0.6, 0.4]];
    let t = [[1.0_f32, 0.0], [1.0, 0.0]];
    let flat = |m: [[f32; 2]; 2]| [m[0][0], m[0][1], m[1][0], m[1][1]];

    assert_eq!(
        BinaryCrossEntropy.value(&y, &t),
        BinaryCrossEntropy.value(&flat(y), &flat(t))
    );
    assert_eq!(
        CategoricalCrossEntropy.value(&y, &t),
        CategoricalCrossEntropy.value(&flat(y), &flat(t))
    );
}

/// Eingabe ist eine Matrix: flachdrücken, dann Dense. Gelernt wird die Summe aller Einträge.
#[test]
fn a_network_can_take_a_matrix_as_input() {
    let mut net = ReshapeLayer::<[[f32; 2]; 2], [f32; 4]>::new()
        .then(DenseLayer::<4, 1>::new([[0.0; 4]], [0.0]));
    let data: Vec<([[f32; 2]; 2], [f32; 1])> = [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ]
    .iter()
    .map(|m| (*m, [m.iter().flatten().sum()]))
    .collect();

    let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.02), 5);
    let before = trainer.evaluate(&net, &data);
    let after = trainer.fit(&mut net, &data, 2000);
    assert!(after < before * 1e-6, "{before} -> {after}");
}

/// Ausgabe ist ein Bild: Vektor rein, Dense, in ein 2×2-Bild falten. Der Verlust wird auf dem
/// Bild berechnet.
#[test]
fn a_network_can_produce_a_matrix_as_output() {
    let mut net = DenseLayer::<1, 4>::new([[0.0]; 4], [0.0; 4])
        .then(ReshapeLayer::<[f32; 4], [[f32; 2]; 2]>::new());
    let target = [[1.0_f32, -1.0], [2.0, 0.5]];
    // Zentrierte Eingaben (−1 und 1), dann lernen Gewicht und Bias unabhängig voneinander.
    let data = [
        ([1.0_f32], target),
        ([-1.0], target.map(|row| row.map(|v| -v))),
    ];

    let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 2);
    trainer.fit(&mut net, &data, 1000);

    let out = net.forward(&[1.0]);
    for (row, expected) in out.iter().zip(target) {
        for (a, b) in row.iter().zip(expected) {
            assert!((a - b).abs() < 1e-3, "{out:?}");
        }
    }
}

/// Bild rein, ReLU auf dem Bild, flachdrücken, Dense: alles hintereinander, nur das Training
/// wird geprüft (der Verlust sinkt).
#[test]
fn a_chain_with_an_image_in_the_middle_trains() {
    let mut net = Block::<Elementwise<Image, Identity>>::new(Identity)
        .then(Block::<Elementwise<Image, Relu>>::new(Relu))
        .then(ReshapeLayer::<Image, [f32; 12]>::new())
        .then(DenseLayer::<12, 1>::new([[0.01; 12]], [0.0]));
    let data = [(image(), [3.0_f32])];

    let mut trainer = Trainer::online(MeanSquaredError, Sgd::new(0.005));
    let before = trainer.evaluate(&net, &data);
    let after = trainer.fit(&mut net, &data, 300);
    assert!(after < before * 1e-3, "{before} -> {after}");
    // Rückwärts bis zur Eingabe: der Gradient hat wieder die Form des Bildes.
    net.forward_mut(&image());
    let grad: &Image = net.backward_mut(&[1.0]);
    assert_eq!(grad[0][0][1], 0.0); // −2 wurde von ReLU abgeschnitten
}
