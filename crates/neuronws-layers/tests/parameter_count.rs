use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::{Chain, LayerExt};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::optimizer::Momentum;
use neuronws_layers::traits::Trainable;

fn visited<M: Trainable>(model: &mut M) -> usize {
    let mut count = 0;
    model.visit_parameters(&mut |_, _| count += 1);
    count
}

#[test]
fn dense_layer_has_weights_plus_biases() {
    assert_eq!(<DenseLayer<2, 3> as Trainable>::PARAMETER_COUNT, 2 * 3 + 3);
    assert_eq!(<DenseLayer<1, 1> as Trainable>::PARAMETER_COUNT, 2);
}

#[test]
fn activation_layer_has_none() {
    assert_eq!(<ActivationLayer<5, Relu> as Trainable>::PARAMETER_COUNT, 0);
}

#[test]
fn chain_adds_up_its_parts() {
    type Net = Chain<DenseLayer<2, 3>, Chain<ActivationLayer<3, Relu>, DenseLayer<3, 1>>>;
    // Dense 2→3: 9, Aktivierung: 0, Dense 3→1: 4
    assert_eq!(<Net as Trainable>::PARAMETER_COUNT, 9 + 4);
}

#[test]
fn the_constant_matches_what_visit_parameters_really_visits() {
    let mut layer = DenseLayer::<3, 2>::new([[0.0; 3]; 2], [0.0; 2]);
    assert_eq!(
        visited(&mut layer),
        <DenseLayer<3, 2> as Trainable>::PARAMETER_COUNT
    );

    let mut net = DenseLayer::<2, 4>::new([[0.0; 2]; 4], [0.0; 4])
        .then(ActivationLayer::<4, _>::new(Relu))
        .then(DenseLayer::<4, 1>::new([[0.0; 4]], [0.0]));
    assert_eq!(visited(&mut net), 4 * 2 + 4 + 4 + 1);
}

#[test]
fn p_can_be_taken_from_the_model_type() {
    type Net = Chain<DenseLayer<2, 3>, DenseLayer<3, 1>>;
    const P: usize = <Net as Trainable>::PARAMETER_COUNT;
    let _momentum = Momentum::<P>::new(0.1, 0.9);
    assert_eq!(P, 13);
}
