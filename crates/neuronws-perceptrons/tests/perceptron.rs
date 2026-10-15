use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::datasets::logic::{AND, NAND, OR, XOR};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::optimizer::{Optimizer, Sgd};
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};
use neuronws_perceptrons::Perceptron;

fn assert_classifies(perceptron: &Perceptron<2>, data: &[([f32; 2], f32)]) {
    for (input, target) in data {
        assert_eq!(perceptron.predict(input), *target, "{input:?}");
    }
}

#[test]
fn predict_applies_the_threshold() {
    let and = Perceptron::<2>::new([1.0, 1.0], -1.5);
    assert_classifies(&and, &AND);
}

#[test]
fn the_threshold_includes_zero() {
    // z = 0 liegt genau auf der Gerade und zählt noch zu "1".
    let p = Perceptron::<1>::new([1.0], -1.0);
    assert_eq!(p.net_input(&[1.0]), 0.0);
    assert_eq!(p.predict(&[1.0]), 1.0);
    assert_eq!(p.predict(&[0.999]), 0.0);
}

#[test]
fn net_input_is_the_weighted_sum_plus_bias() {
    let p = Perceptron::<3>::new([2.0, -1.0, 0.5], 0.25);
    assert_eq!(p.net_input(&[1.0, 2.0, 4.0]), 2.0 - 2.0 + 2.0 + 0.25);
}

#[test]
fn a_perceptron_can_be_made_at_compile_time() {
    const AND_GATE: Perceptron<2> = Perceptron::new([1.0, 1.0], -1.5);
    static OR_GATE: Perceptron<2> = Perceptron::new([1.0, 1.0], -0.5);
    assert_classifies(&AND_GATE, &AND);
    assert_classifies(&OR_GATE, &OR);
}

#[test]
fn learn_follows_the_perceptron_rule() {
    // y = Step(0) = 1, Ziel 0, Fehler = 1: w ← w − η x, b ← b − η
    let mut p = Perceptron::<2>::new([0.0, 0.0], 0.0);
    let error = p.learn(&[1.0, 1.0], 0.0, 0.5);

    assert_eq!(error, 1.0);
    assert_eq!(p.weights(), &[-0.5, -0.5]);
    assert_eq!(p.bias(), -0.5);
}

#[test]
fn a_missed_one_pulls_the_weights_towards_the_input() {
    // y = Step(−1) = 0, Ziel 1, Fehler = −1: w ← w + η x, b ← b + η
    let mut p = Perceptron::<2>::new([0.0, 0.0], -1.0);
    let error = p.learn(&[2.0, 3.0], 1.0, 0.1);
    assert_eq!(error, -1.0);
    assert_eq!(p.weights(), &[0.2, 0.3]);
    assert_eq!(p.bias(), -0.9);
}

#[test]
fn learn_does_nothing_when_the_prediction_is_right() {
    let mut p = Perceptron::<2>::new([1.0, 1.0], -1.5);
    let error = p.learn(&[1.0, 1.0], 1.0, 0.5);

    assert_eq!(error, 0.0);
    assert_eq!(p.weights(), &[1.0, 1.0]);
    assert_eq!(p.bias(), -1.5);
}

#[test]
fn the_classical_rule_equals_one_gradient_step() {
    // Dieselbe Änderung über die Bausteine der Bibliothek: Dense mit einem Neuron, der Fehler
    // `y − t` als Gradient am Ausgang, ein SGD-Schritt. Bit für Bit gleich.
    let examples = [
        ([0.3, -1.2], 1.0),
        ([2.0, 0.5], 0.0),
        ([-0.7, 0.9], 1.0),
        ([1.5, 1.5], 0.0),
        ([-2.0, -0.1], 1.0),
    ];
    let learning_rate = 0.37;
    let mut classical = Perceptron::<2>::new([0.2, -0.4], 0.1);
    let mut dense = classical.into_dense();

    for (input, target) in examples {
        let output = classical.predict(&input);
        classical.learn(&input, target, learning_rate);

        let z = dense.forward_mut(&input)[0];
        let y = if z >= 0.0 { 1.0 } else { 0.0 };
        assert_eq!(y, output, "beide sehen dieselbe Antwort");
        dense.backward_mut(&[y - target]);
        Sgd::new(learning_rate).step(&mut dense, 1);

        assert_eq!(dense.weights()[0], *classical.weights());
        assert_eq!(dense.bias()[0], classical.bias());
    }
}

#[test]
fn fit_learns_and_or_and_nand() {
    for data in [&AND, &OR, &NAND] {
        let mut p = Perceptron::<2>::new([0.0, 0.0], 0.0);
        assert!(p.fit(data, 100, 0.1).is_some());
        assert_classifies(&p, data);
        assert_eq!(p.mistakes(data), 0);
    }
}

#[test]
fn fit_reports_zero_epochs_when_the_first_round_is_already_perfect() {
    let mut p = Perceptron::<2>::new([1.0, 1.0], -1.5);
    assert_eq!(p.fit(&AND, 10, 0.1), Some(0));
}

#[test]
fn the_learning_rate_only_scales_the_result_when_starting_from_zero() {
    // Von 0 aus ändert eine andere Lernrate nur den Maßstab der Gewichte, nicht die Gerade.
    let mut small = Perceptron::<2>::new([0.0, 0.0], 0.0);
    let mut big = Perceptron::<2>::new([0.0, 0.0], 0.0);
    small.fit(&OR, 100, 0.1);
    big.fit(&OR, 100, 10.0);
    for (input, _) in OR {
        assert_eq!(small.predict(&input), big.predict(&input));
    }
}

#[test]
fn fit_cannot_learn_xor() {
    // XOR ist nicht linear trennbar: Egal wie lange man lernt, es bleiben Fehler.
    let mut p = Perceptron::<2>::new([0.0, 0.0], 0.0);
    assert_eq!(p.fit(&XOR, 1000, 0.1), None);
}

#[test]
fn no_single_perceptron_gets_xor_right_but_every_line_gets_at_least_two_of_four() {
    // Eine Gerade, die alle vier Punkte richtig trennt, gibt es nicht. Wir probieren ein Raster
    // von Geraden aus und zählen die Fehler.
    let mut best = usize::MAX;
    for a in -10..=10 {
        for b in -10..=10 {
            for c in -20..=20 {
                let p = Perceptron::<2>::new([a as f32 * 0.5, b as f32 * 0.5], c as f32 * 0.25);
                best = best.min(p.mistakes(&XOR));
            }
        }
    }
    assert_eq!(best, 1, "höchstens drei von vier, aber nie vier");
}

#[test]
fn xor_needs_two_layers_and_each_part_can_be_learned_by_the_perceptron_rule() {
    // XOR = (x ODER y) UND NICHT (x UND y) = OR UND NAND.
    let mut or = Perceptron::<2>::new([0.0, 0.0], 0.0);
    let mut nand = Perceptron::<2>::new([0.0, 0.0], 0.0);
    assert!(or.fit(&OR, 100, 0.1).is_some());
    assert!(nand.fit(&NAND, 100, 0.1).is_some());

    // Die dritte Stufe bekommt die Antworten der ersten beiden als Eingabe und lernt UND.
    let hidden = |input: &[f32; 2]| [or.predict(input), nand.predict(input)];
    let second_stage: Vec<([f32; 2], f32)> = XOR
        .iter()
        .map(|(input, target)| (hidden(input), *target))
        .collect();
    let mut and = Perceptron::<2>::new([0.0, 0.0], 0.0);
    assert!(and.fit(&second_stage, 100, 0.1).is_some());

    for (input, target) in XOR {
        assert_eq!(and.predict(&hidden(&input)), target, "{input:?}");
    }
}

#[test]
fn perceptron_chains_like_any_layer() {
    let and = Perceptron::<2>::new([1.0, 1.0], -1.5);
    let net = and.then(ActivationLayer::<1, _>::new(Relu));
    assert_eq!(net.forward(&[1.0, 1.0]), [1.0]);
    assert_eq!(net.forward(&[1.0, 0.0]), [0.0]);
}

#[test]
fn as_a_dense_layer_with_step_it_calculates_the_same() {
    use neuronws_layers::activation::Step;
    use neuronws_layers::activation_layer::ActivationLayer;

    let p = Perceptron::<2>::new([0.7, -1.3], 0.2);
    let dense = p.into_dense().then(ActivationLayer::<1, _>::new(Step));
    for input in [[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [-1.0, -1.0], [0.3, 0.1]] {
        assert_eq!(dense.forward(&input), p.forward(&input), "{input:?}");
    }
    assert_eq!(<DenseLayer<2, 1> as Trainable>::PARAMETER_COUNT, 3);
}

#[test]
fn perceptron_works_with_f64() {
    let mut p = Perceptron::<2, f64>::new([0.0, 0.0], 0.0);
    let data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 0.0),
        ([1.0, 0.0], 0.0),
        ([1.0, 1.0], 1.0),
    ];
    assert!(p.fit(&data, 100, 0.1).is_some());
    for (input, target) in &data {
        assert_eq!(p.predict(input), *target);
    }
}

#[test]
fn it_can_be_cloned_compared_and_printed() {
    let p = Perceptron::<2>::new([1.0, 2.0], 3.0);
    let q = p;
    assert_eq!(p, q);
    assert_ne!(p, Perceptron::<2>::new([1.0, 2.0], 4.0));
    assert!(format!("{p:?}").starts_with("Perceptron"));
}
