//! Rekurrente Schichten, Embedding und die Bausteine rund um Folgen.

// Die Vergleiche laufen über Zeitschritte und Zustände mehrerer Felder.
#![allow(clippy::needless_range_loop)]

use neuronws_layers::activation::HardTanh;
use neuronws_layers::gradcheck::assert_gradients_match;
use neuronws_layers::recurrent::{Gate, Rnn, RnnLayer};
use neuronws_layers::sequence::{
    Embedding, EmbeddingLayer, LastStep, LastStepLayer, MeanOverSteps, MeanOverStepsLayer,
};
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};
use neuronws_layers::values;

/// Zahlen, die sich nicht wiederholen und nicht zu groß sind.
fn pseudo(index: usize) -> f64 {
    (((index * 37 + 11) % 23) as f64 / 11.0 - 1.0) * 0.6
}

#[cfg(any(feature = "std", feature = "libm"))]
fn gate<const IN: usize, const H: usize>(seed: usize) -> Gate<IN, H, f64> {
    (
        (
            values::from_fn(|i| pseudo(i + seed)),
            values::from_fn(|i| pseudo(i + seed + 31)),
        ),
        values::from_fn(|i| pseudo(i + seed + 63)),
    )
}

fn sequence<const STEPS: usize, const D: usize>(seed: usize) -> [[f64; D]; STEPS] {
    values::from_fn(|i| pseudo(i * 5 + seed))
}

mod rnn {
    use super::*;
    use neuronws_layers::block::Operation;

    #[test]
    fn it_follows_the_recurrence() {
        // h[t] = clamp(1.0·x[t] + 0.5·h[t−1], −1, 1), mit Zahlen, die in Binärdarstellung exakt sind
        let layer = RnnLayer::<3, 1, 1, HardTanh>::from_parts(
            Rnn::new(HardTanh),
            (([[1.0]], [[0.5]]), [0.0]),
        );
        assert_eq!(
            layer.forward(&[[0.5], [0.5], [0.5]]),
            [[0.5], [0.75], [0.875]]
        );
        // und es sättigt bei 1
        assert_eq!(
            layer.forward(&[[2.0], [0.0], [0.0]]),
            [[1.0], [0.5], [0.25]]
        );
    }

    #[test]
    fn the_state_carries_information_forward_in_time() {
        let layer = RnnLayer::<3, 1, 1, HardTanh>::from_parts(
            Rnn::new(HardTanh),
            (([[1.0]], [[0.5]]), [0.0]),
        );
        let a = layer.forward(&[[0.2], [0.0], [0.0]]);
        let b = layer.forward(&[[0.8], [0.0], [0.0]]);
        // Der erste Wert wirkt noch in Schritt 3 nach.
        assert_ne!(a[2], b[2]);

        // Ohne Rückkopplung (Wh = 0) ist jeder Schritt unabhängig.
        let forgetful = RnnLayer::<3, 1, 1, HardTanh>::from_parts(
            Rnn::new(HardTanh),
            (([[1.0]], [[0.0]]), [0.0]),
        );
        assert_eq!(
            forgetful.forward(&[[0.2], [0.0], [0.0]])[2],
            forgetful.forward(&[[0.8], [0.0], [0.0]])[2]
        );
    }

    #[test]
    fn the_gradient_flows_back_through_the_steps() {
        let mut layer = RnnLayer::<4, 1, 1, HardTanh>::from_parts(
            Rnn::new(HardTanh),
            (([[1.0]], [[0.5]]), [0.0]),
        );
        layer.forward_mut(&[[0.1]; 4]);
        // Nur der letzte Schritt bekommt Gradient ...
        let grad = *layer.backward_mut(&[[0.0], [0.0], [0.0], [1.0]]);
        // ... trotzdem erreicht er den ersten Schritt, abgeschwächt um 0.5 pro Schritt.
        assert_eq!(grad, [[0.125], [0.25], [0.5], [1.0]]);

        let mut forgetful = RnnLayer::<4, 1, 1, HardTanh>::from_parts(
            Rnn::new(HardTanh),
            (([[1.0]], [[0.0]]), [0.0]),
        );
        forgetful.forward_mut(&[[0.1]; 4]);
        let grad = *forgetful.backward_mut(&[[0.0], [0.0], [0.0], [1.0]]);
        assert_eq!(grad, [[0.0], [0.0], [0.0], [1.0]]);
    }

    #[test]
    fn the_parameter_count_is_h_times_in_plus_h_plus_one() {
        assert_eq!(
            <RnnLayer<10, 3, 5, HardTanh> as Trainable>::PARAMETER_COUNT,
            5 * (3 + 5 + 1)
        );
        // unabhängig von der Folgenlänge
        assert_eq!(
            <RnnLayer<2, 3, 5, HardTanh> as Trainable>::PARAMETER_COUNT,
            <RnnLayer<200, 3, 5, HardTanh> as Trainable>::PARAMETER_COUNT
        );
    }

    #[test]
    fn the_gradients_are_correct_with_hard_tanh_away_from_the_kinks() {
        // kleine Gewichte und Eingaben halten alle Vorstufen im linearen Bereich (−1, 1)
        let small: Gate<2, 3, f64> = (
            (
                values::from_fn(|i| 0.3 * pseudo(i)),
                values::from_fn(|i| 0.3 * pseudo(i + 9)),
            ),
            values::from_fn(|i| 0.1 * pseudo(i + 4)),
        );
        let input: [[f64; 2]; 4] = values::from_fn(|i| 0.5 * pseudo(i * 3));
        assert_gradients_match(
            &Rnn::<4, 2, 3, _, f64>::new(HardTanh),
            small,
            input,
            1e-6,
            1e-7,
        );
    }

    #[test]
    fn it_works_for_f32_too() {
        let op = Rnn::<2, 1, 1, _, f32>::new(HardTanh);
        let out = op.forward(&(([[1.0]], [[0.5]]), [0.0]), &[[0.2], [0.2]]);
        assert_eq!(out, [[0.2], [0.3]]);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    mod smooth {
        use super::*;
        use neuronws_layers::activation::Tanh;

        #[test]
        fn the_gradients_are_correct_with_tanh() {
            assert_gradients_match(
                &Rnn::<5, 2, 3, _, f64>::new(Tanh),
                gate::<2, 3>(1),
                sequence::<5, 2>(2),
                1e-6,
                1e-6,
            );
            // kürzeste Folge, kleinster Zustand
            assert_gradients_match(
                &Rnn::<1, 1, 1, _, f64>::new(Tanh),
                gate::<1, 1>(3),
                sequence::<1, 1>(4),
                1e-6,
                1e-6,
            );
            assert_gradients_match(
                &Rnn::<3, 2, 2, _, f32>::new(Tanh),
                (
                    ([[0.3_f32, -0.2], [0.1, 0.4]], [[0.2, 0.1], [-0.3, 0.2]]),
                    [0.0, 0.1],
                ),
                [[0.5, -0.5], [0.2, 0.3], [-0.4, 0.1]],
                1e-2,
                5e-3,
            );
        }

        #[test]
        fn the_outputs_stay_between_minus_one_and_one() {
            let op = Rnn::<20, 2, 4, _, f64>::new(Tanh);
            let big: [[f64; 2]; 20] = values::from_fn(|i| 50.0 * pseudo(i));
            let out = op.forward(&gate::<2, 4>(8), &big);
            assert!(out.iter().flatten().all(|h| h.abs() <= 1.0));
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
mod gated {
    use super::*;
    use neuronws_layers::block::Operation;
    use neuronws_layers::init::Zeros;
    use neuronws_layers::random::Rng;
    use neuronws_layers::recurrent::{Gru, GruLayer, Lstm, LstmLayer};

    type GruParams<const IN: usize, const H: usize> =
        ((Gate<IN, H, f64>, Gate<IN, H, f64>), Gate<IN, H, f64>);
    type LstmParams<const IN: usize, const H: usize> = (
        (Gate<IN, H, f64>, Gate<IN, H, f64>),
        (Gate<IN, H, f64>, Gate<IN, H, f64>),
    );

    fn gru_params<const IN: usize, const H: usize>() -> GruParams<IN, H> {
        ((gate(1), gate(40)), gate(80))
    }

    fn lstm_params<const IN: usize, const H: usize>() -> LstmParams<IN, H> {
        ((gate(1), gate(40)), (gate(80), gate(120)))
    }

    #[test]
    fn the_gru_gradients_are_correct() {
        assert_gradients_match(
            &Gru::<5, 2, 3, f64>::new(),
            gru_params::<2, 3>(),
            sequence::<5, 2>(2),
            1e-6,
            1e-6,
        );
        assert_gradients_match(
            &Gru::<1, 1, 1, f64>::new(),
            gru_params::<1, 1>(),
            sequence::<1, 1>(4),
            1e-6,
            1e-6,
        );
        assert_gradients_match(
            &Gru::<4, 3, 2, f64>::new(),
            gru_params::<3, 2>(),
            sequence::<4, 3>(7),
            1e-6,
            1e-6,
        );
    }

    #[test]
    fn the_lstm_gradients_are_correct() {
        assert_gradients_match(
            &Lstm::<5, 2, 3, f64>::new(),
            lstm_params::<2, 3>(),
            sequence::<5, 2>(2),
            1e-6,
            1e-6,
        );
        assert_gradients_match(
            &Lstm::<1, 1, 1, f64>::new(),
            lstm_params::<1, 1>(),
            sequence::<1, 1>(4),
            1e-6,
            1e-6,
        );
        assert_gradients_match(
            &Lstm::<4, 3, 2, f64>::new(),
            lstm_params::<3, 2>(),
            sequence::<4, 3>(7),
            1e-6,
            1e-6,
        );
    }

    #[test]
    fn the_gated_layers_also_work_for_f32() {
        let layer = GruLayer::<3, 2, 4>::initialized(Gru::new(), &Zeros, &mut Rng::new(1));
        let out = layer.forward(&[[1.0, -1.0], [0.5, 0.5], [0.0, 2.0]]);
        assert!(out.iter().flatten().all(|h| h.is_finite()));
        let layer = LstmLayer::<3, 2, 4>::initialized(Lstm::new(), &Zeros, &mut Rng::new(1));
        let out = layer.forward(&[[1.0, -1.0], [0.5, 0.5], [0.0, 2.0]]);
        assert!(out.iter().flatten().all(|h| h.is_finite()));
    }

    #[test]
    fn the_parameter_counts() {
        assert_eq!(
            <GruLayer<10, 3, 5> as Trainable>::PARAMETER_COUNT,
            3 * 5 * (3 + 5 + 1)
        );
        assert_eq!(
            <LstmLayer<10, 3, 5> as Trainable>::PARAMETER_COUNT,
            4 * 5 * (3 + 5 + 1)
        );
    }

    #[test]
    fn the_hidden_state_is_bounded() {
        let big: [[f64; 2]; 15] = values::from_fn(|i| 30.0 * pseudo(i));
        let gru = Gru::<15, 2, 3, f64>::new().forward(&gru_params::<2, 3>(), &big);
        let lstm = Lstm::<15, 2, 3, f64>::new().forward(&lstm_params::<2, 3>(), &big);
        assert!(gru.iter().flatten().all(|h| h.abs() <= 1.0));
        assert!(lstm.iter().flatten().all(|h| h.abs() <= 1.0));
    }

    #[test]
    fn a_closed_gru_update_gate_writes_nothing() {
        // Update-Tor z ≈ 1 (großer Bias): h = h_alt, und h_alt ist am Anfang 0.
        let mut params = gru_params::<1, 2>();
        params.0.1 = ((([[0.0]; 2]), [[0.0; 2]; 2]), [50.0; 2]);
        let out = Gru::<4, 1, 2, f64>::new().forward(&params, &[[1.0]; 4]);
        assert!(out.iter().flatten().all(|h| h.abs() < 1e-12));
    }

    #[test]
    fn an_open_gru_update_gate_forgets_the_past() {
        // z ≈ 0: h = n. Damit n nicht über den Rücksetz-Anteil `r ⊙ h_alt` doch von früher
        // abhängt, ist auch das Rücksetz-Tor zu (r ≈ 0).
        let mut params = gru_params::<1, 2>();
        params.0.1 = ((([[0.0]; 2]), [[0.0; 2]; 2]), [-50.0; 2]);
        params.0.0 = ((([[0.0]; 2]), [[0.0; 2]; 2]), [-50.0; 2]);
        let op = Gru::<3, 1, 2, f64>::new();
        let a = op.forward(&params, &[[5.0], [0.0], [0.3]]);
        let b = op.forward(&params, &[[-5.0], [0.0], [0.3]]);
        for j in 0..2 {
            assert!((a[2][j] - b[2][j]).abs() < 1e-12);
        }
        // Gegenprobe mit offenem Gedächtnis: Hier wirkt die erste Eingabe nach.
        let remembering = gru_params::<1, 2>();
        let c = op.forward(&remembering, &[[5.0], [0.0], [0.3]]);
        let d = op.forward(&remembering, &[[-5.0], [0.0], [0.3]]);
        assert!((0..2).any(|j| (c[2][j] - d[2][j]).abs() > 1e-6));
    }

    #[test]
    fn the_lstm_starts_by_remembering() {
        // Der Bias des Vergiss-Tors startet bei 1, alle anderen bei 0.
        let layer = LstmLayer::<3, 2, 4>::initialized(Lstm::new(), &Zeros, &mut Rng::new(1));
        let ((input_gate, forget_gate), (candidate, output_gate)) = layer.params();
        assert_eq!(input_gate.1, [0.0; 4]);
        assert_eq!(forget_gate.1, [1.0; 4]);
        assert_eq!(candidate.1, [0.0; 4]);
        assert_eq!(output_gate.1, [0.0; 4]);
    }

    #[test]
    fn the_lstm_cell_keeps_a_value_over_many_steps() {
        // Eingabe-Tor nur im ersten Schritt offen (Bias − 50 + 100·x), Vergiss-Tor immer offen.
        // Die Zelle behält dann den Wert aus Schritt 0 bis zum Ende.
        let zero_weights =
            |bias: f64, x_weight: f64| -> Gate<1, 1, f64> { (([[x_weight]], [[0.0]]), [bias]) };
        let params: LstmParams<1, 1> = (
            (zero_weights(-50.0, 100.0), zero_weights(50.0, 0.0)), // i öffnet bei x = 1, f ≈ 1
            (zero_weights(0.0, 1.0), zero_weights(50.0, 0.0)),     // g = tanh(x), o ≈ 1
        );
        let mut input = [[0.0]; 8];
        input[0] = [1.0];
        let out = Lstm::<8, 1, 1, f64>::new().forward(&params, &input);
        let first = out[0][0];
        assert!(first > 0.5, "{first}");
        for step in 1..8 {
            assert!((out[step][0] - first).abs() < 1e-6, "Schritt {step}");
        }
    }

    #[test]
    fn the_layers_can_be_chained_with_last_step() {
        use neuronws_layers::chain::LayerExt;
        use neuronws_layers::dense_layer::DenseLayer;
        let net = GruLayer::<4, 2, 3>::initialized(Gru::new(), &Zeros, &mut Rng::new(1))
            .then(LastStepLayer::<4, 3>::new())
            .then(DenseLayer::<3, 1>::new([[1.0, 1.0, 1.0]], [0.0]));
        let out = net.forward(&[[0.1, 0.2]; 4]);
        assert!(out[0].is_finite());
    }

    #[test]
    fn zero_initial_weights_give_a_zero_gru_state() {
        let layer = GruLayer::<3, 2, 2>::initialized(Gru::new(), &Zeros, &mut Rng::new(1));
        // Bei lauter Nullen ist der Kandidat tanh(0) = 0, also bleibt der Zustand 0.
        assert_eq!(layer.forward(&[[1.0, 2.0]; 3]), [[0.0; 2]; 3]);
    }
}

mod embedding {
    use super::*;
    use neuronws_layers::block::Operation;

    fn table() -> [[f64; 2]; 4] {
        [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]
    }

    #[test]
    fn it_looks_up_a_vector_per_token() {
        let layer = EmbeddingLayer::<4, 2, 3, f64>::new(table());
        assert_eq!(
            layer.forward(&[3.0, 0.0, 3.0]),
            [[3.0, 3.5], [0.0, 0.5], [3.0, 3.5]]
        );
    }

    #[test]
    fn numbers_are_rounded_to_the_nearest_token() {
        let layer = EmbeddingLayer::<4, 2, 3, f64>::new(table());
        assert_eq!(
            layer.forward(&[0.4, 1.6, 2.49]),
            [[0.0, 0.5], [2.0, 2.5], [2.0, 2.5]]
        );
    }

    #[test]
    fn the_gradients_of_repeated_tokens_add_up() {
        let mut layer = EmbeddingLayer::<4, 2, 3, f64>::new(table());
        layer.forward_mut(&[2.0, 1.0, 2.0]);
        let grad_ids = *layer.backward_mut(&[[1.0, 10.0], [100.0, 1000.0], [0.5, 0.25]]);
        assert_eq!(grad_ids, [0.0; 3]);
        assert_eq!(layer.grad_table()[2], [1.5, 10.25]);
        assert_eq!(layer.grad_table()[1], [100.0, 1000.0]);
        assert_eq!(layer.grad_table()[0], [0.0, 0.0]);
        assert_eq!(layer.grad_table()[3], [0.0, 0.0]);
        assert_eq!(layer.table(), &table());
    }

    #[test]
    fn the_gradient_with_respect_to_the_table_is_correct() {
        assert_gradients_match(
            &Embedding::<4, 2, 3, f64>::new(),
            table(),
            [2.0, 1.0, 2.0],
            1e-6,
            1e-7,
        );
    }

    #[test]
    #[should_panic(expected = "Token-Nummer 4 liegt nicht in 0..4")]
    fn a_token_beyond_the_vocabulary_is_refused() {
        let layer = EmbeddingLayer::<4, 2, 1, f64>::new(table());
        let _ = layer.forward(&[4.0]);
    }

    #[test]
    #[should_panic(expected = "liegt nicht in 0..4")]
    fn a_negative_token_is_refused() {
        let layer = EmbeddingLayer::<4, 2, 1, f64>::new(table());
        let _ = layer.forward(&[-1.0]);
    }

    #[test]
    fn it_has_one_row_of_parameters_per_token() {
        assert_eq!(
            <EmbeddingLayer<1000, 16, 20> as Trainable>::PARAMETER_COUNT,
            1000 * 16
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn start_values_come_from_the_initializer() {
        use neuronws_layers::block::Block;
        use neuronws_layers::init::Uniform;
        use neuronws_layers::random::Rng;
        let layer = Block::initialized(
            Embedding::<10, 4, 3>::new(),
            &Uniform(0.1),
            &mut Rng::new(1),
        );
        assert!(layer.table().iter().flatten().all(|v| v.abs() <= 0.1));
        assert!(layer.table().iter().flatten().any(|&v| v != 0.0));
    }

    #[test]
    fn it_learns_which_tokens_belong_together() {
        use neuronws_layers::chain::LayerExt;
        use neuronws_layers::dense_layer::DenseLayer;
        use neuronws_layers::loss::MeanSquaredError;
        use neuronws_layers::optimizer::Sgd;
        use neuronws_layers::random::Rng;
        use neuronws_layers::trainer::Trainer;

        // Tokens 0 und 1 gehören zu Klasse 0, Tokens 2 und 3 zu Klasse 1, Eingabe ist ein Token.
        let mut rng = Rng::new(3);
        let mut net = EmbeddingLayer::<4, 2, 1>::new([[0.0; 2]; 4])
            .then(MeanOverStepsLayer::<1, 2>::new())
            .then(DenseLayer::<2, 1>::random(&mut rng, 0.5));
        // Startwerte der Tabelle: klein, zufällig
        net.visit_parameters(&mut |p, _| *p = 0.1 * rng.uniform(-1.0, 1.0));
        let data = [
            ([0.0_f32], [0.0_f32]),
            ([1.0], [0.0]),
            ([2.0], [1.0]),
            ([3.0], [1.0]),
        ];
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.3), 4);
        trainer.fit(&mut net, &data, 400);
        for (token, target) in &data {
            let out = net.forward(token)[0];
            assert!((out - target[0]).abs() < 0.2, "Token {token:?}: {out}");
        }
    }

    #[test]
    fn the_operation_does_not_change_anything_by_itself() {
        let op = Embedding::<4, 2, 2, f64>::new();
        let a = op.forward(&table(), &[1.0, 2.0]);
        let b = op.forward(&table(), &[1.0, 2.0]);
        assert_eq!(a, b);
    }
}

mod pooling_over_time {
    use super::*;

    #[test]
    fn last_step_takes_the_last_row() {
        let layer = LastStepLayer::<3, 2>::new();
        assert_eq!(
            layer.forward(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            [5.0, 6.0]
        );
    }

    #[test]
    fn last_step_gives_gradient_only_to_the_last_row() {
        let mut layer = LastStepLayer::<3, 2>::new();
        layer.forward_mut(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        assert_eq!(
            *layer.backward_mut(&[0.5, -1.0]),
            [[0.0, 0.0], [0.0, 0.0], [0.5, -1.0]]
        );
    }

    #[test]
    fn mean_over_steps_averages_the_rows() {
        let layer = MeanOverStepsLayer::<4, 2>::new();
        assert_eq!(
            layer.forward(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            [4.0, 5.0]
        );
    }

    #[test]
    fn mean_over_steps_shares_the_gradient_evenly() {
        let mut layer = MeanOverStepsLayer::<4, 2>::new();
        layer.forward_mut(&[[0.0; 2]; 4]);
        assert_eq!(*layer.backward_mut(&[2.0, -4.0]), [[0.5, -1.0]; 4]);
    }

    #[test]
    fn the_gradients_are_correct() {
        assert_gradients_match(
            &LastStep::<3, 2, f64>::new(),
            (),
            sequence::<3, 2>(1),
            1e-6,
            1e-7,
        );
        assert_gradients_match(
            &MeanOverSteps::<3, 2, f64>::new(),
            (),
            sequence::<3, 2>(1),
            1e-6,
            1e-7,
        );
    }

    #[test]
    fn they_have_no_parameters() {
        assert_eq!(<LastStepLayer<5, 3> as Trainable>::PARAMETER_COUNT, 0);
        assert_eq!(<MeanOverStepsLayer<5, 3> as Trainable>::PARAMETER_COUNT, 0);
    }
}
