//! Ein ganzes Faltungsnetz lernt, Linien in verrauschten Bildern zu unterscheiden.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::adam::AdamW;
use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::conv::Conv2dValid;
use neuronws_layers::datasets::lines;
use neuronws_layers::dense_layer::Linear;
use neuronws_layers::init::{He, Xavier};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::metrics::accuracy;
use neuronws_layers::neurons::neurons;
use neuronws_layers::pooling::MaxPool2d;
use neuronws_layers::random::Rng;
use neuronws_layers::reshape::Reshape;
use neuronws_layers::softmax_layer::Softmax;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::Trainable;
use neuronws_layers::values::Image;

#[test]
fn a_small_cnn_tells_vertical_horizontal_and_diagonal_lines_apart() {
    let mut rng = Rng::new(7);
    let train = lines::<120>(&mut rng);
    let test = lines::<300>(&mut rng);

    // Faltung und ReLU in einem Aufruf: `neurons` kennt keine Formen und geht deshalb auch für Bilder.
    let mut net = neurons(
        Conv2dValid::<1, 4, 6, 6, 3, 3, 4, 4>::new(),
        &He,
        Elementwise::new(Relu),
        &mut rng,
    )
    .then(Block::from_parts(MaxPool2d::<4, 4, 4, 2, 2, 2>::new(), ()))
    .then(Block::from_parts(
        Reshape::<Image<4, 2, 2>, [f32; 16]>::new(),
        (),
    ))
    .then(neurons(
        Linear::<16, 3>::new(),
        &Xavier,
        Softmax::<3>::new(),
        &mut rng,
    ));

    let mut trainer = Trainer::new(
        CategoricalCrossEntropy,
        AdamW::for_model(&net, 0.01, 0.001),
        8,
    );
    let before = trainer.evaluate(&net, &test);
    for _ in 0..40 {
        trainer.epoch(&mut net, &train);
    }
    let after = trainer.evaluate(&net, &test);

    let hits: f32 = accuracy(&net, &test);
    assert!(after < before * 0.05, "Verlust {before} -> {after}");
    assert!(hits >= 0.95, "Trefferquote {hits}");
}

#[test]
fn the_cnn_has_as_few_parameters_as_expected() {
    type Net = neuronws_layers::chain::Chain<
        neuronws_layers::conv::Conv2dValidLayer<1, 4, 6, 6, 3, 3, 4, 4>,
        neuronws_layers::dense_layer::DenseLayer<16, 3>,
    >;
    // Faltung 4·(3·3) + 4, Dense 16·3 + 3
    assert_eq!(<Net as Trainable>::PARAMETER_COUNT, 40 + 51);
}
