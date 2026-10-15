//! Ein ganzes Faltungsnetz lernt, Linien in verrauschten Bildern zu unterscheiden.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::adam::AdamW;
use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::conv::Conv2d;
use neuronws_layers::dense_layer::Linear;
use neuronws_layers::init::{He, Xavier};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::neurons::neurons;
use neuronws_layers::pooling::MaxPool2d;
use neuronws_layers::random::Rng;
use neuronws_layers::reshape::Reshape;
use neuronws_layers::softmax_layer::Softmax;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{ForwardPath, Trainable};
use neuronws_layers::values::Image;

type Sample = (Image<1, 6, 6>, [f32; 3]);

fn make_image(rng: &mut Rng, class: usize) -> Sample {
    let mut image = [[[0.0_f32; 6]; 6]];
    let at = rng.next_below(6);
    for i in 0..6 {
        match class {
            0 => image[0][i][at] = 1.0,
            1 => image[0][at][i] = 1.0,
            _ if at.is_multiple_of(2) => image[0][i][i] = 1.0,
            _ => image[0][i][5 - i] = 1.0,
        }
    }
    for value in image[0].iter_mut().flatten() {
        *value += rng.uniform(-0.2, 0.2);
    }
    let mut target = [0.0; 3];
    target[class] = 1.0;
    (image, target)
}

fn make_data(rng: &mut Rng, per_class: usize) -> Vec<Sample> {
    let mut data: Vec<Sample> = (0..per_class * 3).map(|i| make_image(rng, i % 3)).collect();
    rng.shuffle(&mut data);
    data
}

fn argmax(values: &[f32; 3]) -> usize {
    (1..3).fold(0, |best, i| if values[i] > values[best] { i } else { best })
}

#[test]
fn a_small_cnn_tells_vertical_horizontal_and_diagonal_lines_apart() {
    let mut rng = Rng::new(7);
    let train = make_data(&mut rng, 40);
    let test = make_data(&mut rng, 100);

    let mut net = Block::initialized(Conv2d::<1, 4, 6, 6, 3, 3, 4, 4>::new(), &He, &mut rng)
        .then(Block::<Elementwise<Image<4, 4, 4>, Relu>>::new(Relu))
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

    let hits = test
        .iter()
        .filter(|(x, t)| argmax(&net.forward(x)) == argmax(t))
        .count();
    assert!(after < before * 0.05, "Verlust {before} -> {after}");
    assert!(hits * 100 >= test.len() * 95, "{hits} von {}", test.len());
}

#[test]
fn the_cnn_has_as_few_parameters_as_expected() {
    type Net = neuronws_layers::chain::Chain<
        neuronws_layers::conv::Conv2dLayer<1, 4, 6, 6, 3, 3, 4, 4>,
        neuronws_layers::dense_layer::DenseLayer<16, 3>,
    >;
    // Faltung 4·(3·3) + 4, Dense 16·3 + 3
    assert_eq!(<Net as Trainable>::PARAMETER_COUNT, 40 + 51);
}
