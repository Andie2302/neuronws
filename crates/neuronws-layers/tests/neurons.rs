//! Schichten aus austauschbaren Teilen zusammenbauen und trainieren.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::{Identity, Relu, Sigmoid, Tanh};
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::Linear;
use neuronws_layers::init::{Constant, He, Initializer, Uniform, Xavier, Zeros};
use neuronws_layers::loss::{CategoricalCrossEntropy, MeanSquaredError};
use neuronws_layers::neurons::neurons;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::softmax_layer::Softmax;
use neuronws_layers::training::train_epoch;
use neuronws_layers::traits::{ForwardPath, Trainable};

const XOR: [([f32; 2], [f32; 1]); 4] = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
];

fn assert_learns_xor(
    mut net: impl neuronws_layers::traits::ForwardPathMut<
        ForwardInput = [f32; 2],
        ForwardOutput = [f32; 1],
    > + neuronws_layers::traits::BackwardPathMut<
        BackwardInput = [f32; 1],
        BackwardOutput = [f32; 2],
    > + ForwardPath<ForwardInput = [f32; 2], ForwardOutput = [f32; 1]>
    + Trainable<Scalar = f32>,
    learning_rate: f32,
) {
    let mut sgd = Sgd::new(learning_rate);
    for _ in 0..5000 {
        train_epoch(&mut net, &mut sgd, &MeanSquaredError, &XOR, 4);
    }
    for (x, [target]) in &XOR {
        let [y] = net.forward(x);
        assert!((y - target).abs() < 0.2, "{x:?}: {y} statt {target}");
    }
}

#[test]
fn a_layer_is_built_from_calculation_initializer_and_activation() {
    let mut rng = Rng::new(1);
    let layer = neurons(
        Linear::<35, 16>::new(),
        &Xavier,
        Elementwise::new(Tanh),
        &mut rng,
    );
    assert_eq!(<_ as ForwardPath>::forward(&layer, &[0.5; 35]).len(), 16);
    assert_eq!(layer_parameter_count(&layer), 35 * 16 + 16);
}

fn layer_parameter_count<M: Trainable>(_: &M) -> usize {
    M::PARAMETER_COUNT
}

#[test]
fn layers_chain_into_a_network_that_learns_xor() {
    for seed in 0..5 {
        let mut rng = Rng::new(seed);
        let net = neurons(
            Linear::<2, 4>::new(),
            &Xavier,
            Elementwise::new(Tanh),
            &mut rng,
        )
        .then(neurons(
            Linear::<4, 1>::new(),
            &Xavier,
            Elementwise::new(Identity),
            &mut rng,
        ));
        assert_learns_xor(net, 0.1);
    }
}

#[test]
fn every_part_can_be_swapped_without_touching_the_rest() {
    // andere Startwerte, andere Aktivierung, gleiche Form
    let mut rng = Rng::new(2);
    let net = neurons(
        Linear::<2, 6>::new(),
        &Uniform(0.8),
        Elementwise::new(Sigmoid),
        &mut rng,
    )
    .then(neurons(
        Linear::<6, 1>::new(),
        &Uniform(0.8),
        Elementwise::new(Identity),
        &mut rng,
    ));
    assert_learns_xor(net, 0.5);

    let mut rng = Rng::new(3);
    let net = neurons(Linear::<2, 8>::new(), &He, Elementwise::new(Relu), &mut rng).then(neurons(
        Linear::<8, 1>::new(),
        &He,
        Elementwise::new(Identity),
        &mut rng,
    ));
    assert_learns_xor(net, 0.05);
}

#[test]
fn the_initializer_decides_the_starting_weights() {
    let mut rng = Rng::new(1);
    let zero = neurons(
        Linear::<2, 3>::new(),
        &Zeros,
        Elementwise::new(Identity),
        &mut rng,
    );
    assert_eq!(zero.forward(&[1.0, 2.0]), [0.0; 3]);

    let constant = neurons(
        Linear::<2, 3>::new(),
        &Constant(0.5),
        Elementwise::new(Identity),
        &mut rng,
    );
    assert_eq!(constant.forward(&[1.0, 2.0]), [1.5; 3]);
}

#[test]
fn softmax_can_be_the_activation_of_a_layer() {
    // Drei Klassen im Plan; ein einziger Layer mit Softmax als Aktivierung genügt hier.
    let points: [([f32; 2], usize); 6] = [
        ([-1.0, -1.0], 0),
        ([-1.2, -0.8], 0),
        ([1.0, -1.0], 1),
        ([1.1, -1.2], 1),
        ([0.0, 1.0], 2),
        ([0.1, 1.3], 2),
    ];
    let one_hot = |class: usize| -> [f32; 3] { core::array::from_fn(|i| f32::from(i == class)) };
    let data: Vec<([f32; 2], [f32; 3])> = points.iter().map(|(x, c)| (*x, one_hot(*c))).collect();

    let mut rng = Rng::new(4);
    let mut net = neurons(
        Linear::<2, 3>::new(),
        &Xavier,
        Softmax::<3>::new(),
        &mut rng,
    );
    let mut sgd = Sgd::new(0.5);
    for _ in 0..300 {
        train_epoch(&mut net, &mut sgd, &CategoricalCrossEntropy, &data, 3);
    }
    for (x, class) in &points {
        let y = net.forward(x);
        let guess = (0..3).max_by(|&a, &b| y[a].total_cmp(&y[b])).unwrap();
        assert_eq!(guess, *class, "{x:?}: {y:?}");
        assert!((y.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}

#[test]
fn it_works_for_f64() {
    let mut rng = Rng::new(1);
    let layer = neurons(
        Linear::<2, 3, f64>::new(),
        &Xavier,
        Elementwise::new(Tanh),
        &mut rng,
    );
    let y = layer.forward(&[1.0_f64, -1.0]);
    assert!(y.iter().all(|v| v.abs() < 1.0));
}

#[test]
fn a_custom_initializer_plugs_in_like_a_built_in_one() {
    struct Alternating;
    impl Initializer for Alternating {
        fn sample<T: neuronws_layers::scalar::Scalar>(
            &self,
            rng: &mut Rng,
            _: usize,
            _: usize,
        ) -> T {
            if rng.next_u32().is_multiple_of(2) {
                T::ONE
            } else {
                -T::ONE
            }
        }
    }
    let mut rng = Rng::new(8);
    let layer = neurons(
        Linear::<2, 2>::new(),
        &Alternating,
        Elementwise::new(Identity),
        &mut rng,
    );
    let [a, b] = layer.forward(&[1.0, 1.0]);
    assert!([-2.0, 0.0, 2.0].contains(&a) && [-2.0, 0.0, 2.0].contains(&b));
}
