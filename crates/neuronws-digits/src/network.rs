use neuronws_layers::activation::Tanh;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::{Chain, LayerExt};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::random::Rng;
use neuronws_layers::softmax_layer::SoftmaxLayer;

use crate::{CLASSES, PIXELS};

/// 35 Pixel → 16 verborgene Neuronen (Tanh) → 10 Ziffern (Softmax).
pub type Network = Chain<
    Chain<
        Chain<DenseLayer<PIXELS, HIDDEN>, ActivationLayer<HIDDEN, Tanh>>,
        DenseLayer<HIDDEN, CLASSES>,
    >,
    SoftmaxLayer<CLASSES>,
>;

pub const HIDDEN: usize = 16;

/// Ein neues Netz mit zufälligen Startgewichten (Xavier).
pub fn new_network(rng: &mut Rng) -> Network {
    DenseLayer::<PIXELS, HIDDEN>::xavier(rng)
        .then(ActivationLayer::<HIDDEN, _>::new(Tanh))
        .then(DenseLayer::<HIDDEN, CLASSES>::xavier(rng))
        .then(SoftmaxLayer::<CLASSES>::new())
}
