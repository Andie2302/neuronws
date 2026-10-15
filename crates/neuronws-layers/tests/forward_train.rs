//! `forward_mut` (Lernen) und `forward` (Einsatz) dürfen verschieden rechnen.

use neuronws_layers::block::{Block, Operation};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::init::Initializer;
use neuronws_layers::random::Rng;
use neuronws_layers::traits::{ForwardPath, ForwardPathMut};

/// Rechnet im Einsatz `2·x`, beim Lernen `3·x` und zählt die Lern-Durchgänge.
#[derive(Debug, Default)]
struct Probe {
    train_calls: usize,
}

impl Operation for Probe {
    type Scalar = f32;
    type Input = [f32; 1];
    type Output = [f32; 1];
    type Params = ();

    fn forward(&self, _: &(), x: &[f32; 1]) -> [f32; 1] {
        [2.0 * x[0]]
    }

    fn forward_train(&mut self, _: &(), x: &[f32; 1]) -> [f32; 1] {
        self.train_calls += 1;
        [3.0 * x[0]]
    }

    fn backward(&self, _: &(), _: &mut (), _: &[f32; 1], _: &[f32; 1], g: &[f32; 1]) -> [f32; 1] {
        *g
    }

    fn init_params<I: Initializer>(&self, _: &I, _: &mut Rng) {}
}

#[test]
fn forward_mut_uses_the_training_path_and_forward_the_inference_path() {
    let mut block = Block::from_parts(Probe::default(), ());

    assert_eq!(block.forward(&[1.0]), [2.0]);
    assert_eq!(block.operation().train_calls, 0);

    assert_eq!(*block.forward_mut(&[1.0]), [3.0]);
    assert_eq!(block.operation().train_calls, 1);

    // Der Einsatz zählt nicht mit.
    assert_eq!(block.forward(&[1.0]), [2.0]);
    assert_eq!(block.operation().train_calls, 1);
}

#[test]
fn the_operation_can_be_changed_through_the_block() {
    let mut block = Block::from_parts(Probe::default(), ());
    block.forward_mut(&[1.0]);
    block.operation_mut().train_calls = 0;
    assert_eq!(block.operation().train_calls, 0);
}

#[test]
fn by_default_training_and_inference_calculate_the_same() {
    let mut layer = DenseLayer::new([[1.0, -2.0]], [0.5]);
    let input = [3.0, 1.0];
    let expected = layer.forward(&input);
    assert_eq!(*layer.forward_mut(&input), expected);
}
