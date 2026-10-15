//! Folgen lernen: Das Netz muss sich den **ersten** Wert merken, bis die Folge zu Ende ist.
#![cfg(any(feature = "std", feature = "libm"))]

use neuronws_layers::activation::Tanh;
use neuronws_layers::adam::AdamW;
use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::init::{Orthogonal, Xavier};
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::random::Rng;
use neuronws_layers::recurrent::{Gru, Lstm, Rnn};
use neuronws_layers::sequence::LastStepLayer;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

const STEPS: usize = 8;
const HIDDEN: usize = 8;

type Sequence = [[f32; 1]; STEPS];
type Sample = (Sequence, [f32; 1]);

/// Eine Folge aus lauter −1 und 1. Das Ziel ist der erste Wert.
fn make_sample(rng: &mut Rng) -> Sample {
    let mut sequence = [[0.0; 1]; STEPS];
    for step in &mut sequence {
        step[0] = if rng.next_f32() < 0.5 { -1.0 } else { 1.0 };
    }
    (sequence, [sequence[0][0]])
}

fn make_data(rng: &mut Rng, n: usize) -> Vec<Sample> {
    (0..n).map(|_| make_sample(rng)).collect()
}

/// Anteil der Folgen, bei denen das Vorzeichen der Ausgabe das richtige ist.
fn sign_accuracy<M>(net: &M, data: &[Sample]) -> f32
where
    M: ForwardPath<ForwardInput = Sequence, ForwardOutput = [f32; 1]>,
{
    let right = data
        .iter()
        .filter(|(x, target)| net.forward(x)[0] * target[0] > 0.0)
        .count();
    right as f32 / data.len() as f32
}

/// Lernt mit AdamW 150 Runden und liefert die Trefferquote auf neuen Folgen.
fn learn<M>(mut net: M, seed: u64) -> (f32, f32)
where
    M: ForwardPath<ForwardInput = Sequence, ForwardOutput = [f32; 1]>
        + ForwardPathMut<ForwardInput = Sequence, ForwardOutput = [f32; 1]>
        + BackwardPathMut<BackwardInput = [f32; 1], BackwardOutput = Sequence>
        + Trainable<Scalar = f32>,
{
    let mut rng = Rng::new(seed);
    let train = make_data(&mut rng, 96);
    let test = make_data(&mut rng, 300);
    let mut trainer = Trainer::new(MeanSquaredError, AdamW::for_model(&net, 0.02, 0.0), 16);
    let before = sign_accuracy(&net, &test);
    for _ in 0..150 {
        trainer.epoch(&mut net, &train);
    }
    (before, sign_accuracy(&net, &test))
}

#[test]
fn an_rnn_learns_to_remember_the_first_value() {
    let mut rng = Rng::new(1);
    let net = Block::initialized(
        Rnn::<STEPS, 1, HIDDEN, _>::new(Tanh),
        &Orthogonal(1.0),
        &mut rng,
    )
    .then(LastStepLayer::<STEPS, HIDDEN>::new())
    .then(DenseLayer::<HIDDEN, 1>::xavier(&mut rng));
    let (before, after) = learn(net, 11);
    assert!(before < 0.7, "vorher {before}");
    assert!(after > 0.95, "nachher {after}");
}

#[test]
fn a_gru_learns_to_remember_the_first_value() {
    let mut rng = Rng::new(2);
    let net = Block::initialized(Gru::<STEPS, 1, HIDDEN>::new(), &Xavier, &mut rng)
        .then(LastStepLayer::<STEPS, HIDDEN>::new())
        .then(DenseLayer::<HIDDEN, 1>::xavier(&mut rng));
    let (before, after) = learn(net, 12);
    assert!(before < 0.7, "vorher {before}");
    assert!(after > 0.95, "nachher {after}");
}

#[test]
fn an_lstm_learns_to_remember_the_first_value() {
    let mut rng = Rng::new(3);
    let net = Block::initialized(Lstm::<STEPS, 1, HIDDEN>::new(), &Xavier, &mut rng)
        .then(LastStepLayer::<STEPS, HIDDEN>::new())
        .then(DenseLayer::<HIDDEN, 1>::xavier(&mut rng));
    let (before, after) = learn(net, 13);
    assert!(before < 0.7, "vorher {before}");
    assert!(after > 0.95, "nachher {after}");
}

#[test]
fn without_memory_the_task_cannot_be_solved() {
    // Ein Netz, das nur den letzten Wert sieht, kann den ersten nicht erraten: Es bleibt beim Raten.
    let mut rng = Rng::new(4);
    let net = LastStepLayer::<STEPS, 1>::new().then(DenseLayer::<1, 1>::xavier(&mut rng));
    let (_, after) = learn(net, 14);
    assert!((0.35..0.65).contains(&after), "{after}");
}
