//! Ein winziger Variational Autoencoder (ohne KL-Term): Fork liefert Mittelwert und Varianz, die
//! Reparametrisierung zieht daraus `z`, und der Gradient fließt trotz Zufall bis zum Anfang zurück.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::dropout::ReparameterizeLayer;
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::residual::Fork;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{ForwardPath, ForwardPathMut, Trainable};

#[test]
fn the_encoder_decoder_learns_to_reconstruct_even_though_z_is_random() {
    let mut rng = Rng::new(4);
    let encoder = Fork::new(
        DenseLayer::<2, 2>::random(&mut rng, 0.5),
        DenseLayer::<2, 2>::new([[0.0; 2]; 2], [-2.0, -2.0]), // anfangs kleine Streuung
    );
    let mut net = encoder
        .then(ReparameterizeLayer::<[f32; 2]>::new(9))
        .then(DenseLayer::<2, 2>::random(&mut rng, 0.5));

    // Punkte auf einer Geraden sollen sich selbst ergeben.
    let data: Vec<([f32; 2], [f32; 2])> = (0..16)
        .map(|i| {
            let t = -1.0 + 0.125 * i as f32;
            ([t, 0.5 * t], [t, 0.5 * t])
        })
        .collect();

    let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 4);
    let before = trainer.evaluate(&net, &data);
    for _ in 0..300 {
        trainer.epoch(&mut net, &data);
    }
    let after = trainer.evaluate(&net, &data);
    assert!(after < before * 0.1, "{before} -> {after}");
}

#[test]
fn at_inference_the_result_is_deterministic_and_while_learning_it_is_not() {
    let mut rng = Rng::new(1);
    let mut net = Fork::new(
        DenseLayer::<2, 2>::random(&mut rng, 0.5),
        DenseLayer::<2, 2>::random(&mut rng, 0.5),
    )
    .then(ReparameterizeLayer::<[f32; 2]>::new(3));

    let x = [0.3, -0.8];
    assert_eq!(net.forward(&x), net.forward(&x));
    let a = *net.forward_mut(&x);
    let b = *net.forward_mut(&x);
    assert_ne!(a, b);
    assert_eq!(parameter_count(&net), 2 * 2 * 2 + 2 + 2);
}

/// Anzahl der lernbaren Parameter eines beliebigen Netzes, ohne den Typ hinzuschreiben.
fn parameter_count<M: Trainable>(_model: &M) -> usize {
    M::PARAMETER_COUNT
}
