//! Kennzahlen, Datenaufteilung, Early Stopping und die erweiterten Trainer-Methoden.

use neuronws_layers::activation::{Identity, Relu};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::data::{shuffled_train_test_split, train_test_split};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::early_stopping::{EarlyStopping, Progress};
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::metrics::{accuracy, argmax, binary_accuracy, confusion_matrix};
use neuronws_layers::optimizer::{HasLearningRate, Sgd};
use neuronws_layers::random::Rng;
use neuronws_layers::schedule::ReduceOnPlateau;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::Trainable;

mod metrics {
    use super::*;

    #[test]
    fn argmax_finds_the_largest_value() {
        assert_eq!(argmax(&[0.1_f32, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[5.0_f64]), 0);
        assert_eq!(argmax(&[-3.0_f32, -1.0, -2.0]), 1);
    }

    #[test]
    fn ties_go_to_the_first_and_nan_never_wins() {
        assert_eq!(argmax(&[2.0_f32, 2.0, 1.0]), 0);
        assert_eq!(argmax(&[1.0_f32, 2.0, 2.0]), 1);
        assert_eq!(argmax(&[f32::NAN, 1.0, 0.0]), 1);
        assert_eq!(argmax(&[1.0_f32, f32::NAN, 0.5]), 0);
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(argmax(&[f32::NAN, -1.0, f32::NAN]), 1);
    }

    #[test]
    fn argmax_works_on_any_shape_in_the_order_of_values() {
        assert_eq!(argmax(&[[0.0_f32, 1.0], [5.0, 2.0]]), 2);
        assert_eq!(argmax(&([1.0_f32, 2.0], [3.0, 9.0, 4.0])), 3);
        assert_eq!(argmax::<f32, ()>(&()), 0);
    }

    fn passthrough() -> ActivationLayer<3, Identity> {
        ActivationLayer::new(Identity)
    }

    #[test]
    fn accuracy_counts_the_right_classes() {
        let data = [
            ([0.9, 0.1, 0.0], [1.0, 0.0, 0.0]),
            ([0.2, 0.7, 0.1], [0.0, 1.0, 0.0]),
            ([0.5, 0.4, 0.1], [0.0, 0.0, 1.0]),
            ([0.1, 0.2, 0.7], [0.0, 0.0, 1.0]),
        ];
        assert_eq!(accuracy(&passthrough(), &data), 0.75);
    }

    #[test]
    fn accuracy_is_one_or_zero_at_the_extremes_and_works_for_f64() {
        let all_right = [
            ([1.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        assert_eq!(
            accuracy(&ActivationLayer::<3, _, f64>::new(Identity), &all_right),
            1.0
        );
        let all_wrong = [([1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0])];
        assert_eq!(accuracy(&passthrough(), &all_wrong), 0.0);
    }

    #[test]
    #[should_panic(expected = "data darf nicht leer sein")]
    fn accuracy_needs_data() {
        let none: [([f32; 3], [f32; 3]); 0] = [];
        let _: f32 = accuracy(&passthrough(), &none);
    }

    #[test]
    fn binary_accuracy_needs_every_output_to_match() {
        let net = ActivationLayer::<2, _>::new(Identity);
        let data = [
            ([0.9, 0.1], [1.0, 0.0]), // beide richtig
            ([0.9, 0.6], [1.0, 0.0]), // zweiter Ausgang falsch
            ([0.2, 0.2], [0.0, 0.0]), // beide richtig
            ([0.5, 0.5], [1.0, 1.0]), // genau auf der Schwelle zählt als ja
        ];
        assert_eq!(binary_accuracy(&net, &data), 0.75);
    }

    #[test]
    fn the_confusion_matrix_counts_true_against_predicted() {
        let net = ActivationLayer::<3, _>::new(Identity);
        let data = [
            ([0.9, 0.1, 0.0], [1.0, 0.0, 0.0]),
            ([0.8, 0.1, 0.1], [1.0, 0.0, 0.0]),
            ([0.1, 0.8, 0.1], [1.0, 0.0, 0.0]), // 0 als 1 erkannt
            ([0.1, 0.1, 0.8], [0.0, 1.0, 0.0]), // 1 als 2 erkannt
            ([0.0, 0.1, 0.9], [0.0, 0.0, 1.0]),
        ];
        let matrix = confusion_matrix::<3, _, _, _, _>(&net, &data);
        assert_eq!(matrix, [[2, 1, 0], [0, 0, 1], [0, 0, 1]]);
        // Die Summe aller Einträge ist die Zahl der Beispiele, die Diagonale die Zahl der Treffer.
        assert_eq!(matrix.iter().flatten().sum::<usize>(), data.len());
        let hits: usize = (0..3).map(|i| matrix[i][i]).sum();
        assert_eq!(hits as f32 / data.len() as f32, accuracy(&net, &data));
    }

    #[test]
    #[should_panic(expected = "passt nicht zu K = 2")]
    fn the_matrix_must_be_big_enough() {
        let net = ActivationLayer::<3, _>::new(Identity);
        let _ = confusion_matrix::<2, f32, _, _, _>(&net, &[([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])]);
    }
}

mod splitting {
    use super::*;

    #[test]
    fn the_test_part_is_taken_from_the_end() {
        let data: [u32; 10] = core::array::from_fn(|i| i as u32);
        let (train, test) = train_test_split(&data, 0.3);
        assert_eq!(train, &[0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(test, &[7, 8, 9]);
    }

    #[test]
    fn the_fraction_is_rounded_and_both_parts_stay_non_empty() {
        let data = [0_u8; 10];
        assert_eq!(train_test_split(&data, 0.25).1.len(), 3); // 2.5 rundet auf 3
        assert_eq!(train_test_split(&data, 0.04).1.len(), 1); // wäre 0, wird 1
        assert_eq!(train_test_split(&data, 0.99).0.len(), 1); // wäre 0 Trainingsdaten, wird 1
        assert_eq!(train_test_split(&[1, 2], 0.5), (&[1][..], &[2][..]));
    }

    #[test]
    #[should_panic(expected = "mindestens 2 Beispiele")]
    fn one_example_cannot_be_split() {
        let _ = train_test_split(&[1], 0.5);
    }

    #[test]
    #[should_panic(expected = "test_fraction muss zwischen 0 und 1 liegen")]
    fn a_fraction_of_zero_is_refused() {
        let _ = train_test_split(&[1, 2, 3], 0.0);
    }

    #[test]
    #[should_panic(expected = "test_fraction muss zwischen 0 und 1 liegen")]
    fn a_fraction_of_one_is_refused() {
        let _ = train_test_split(&[1, 2, 3], 1.0);
    }

    #[test]
    fn shuffling_keeps_all_examples_and_changes_the_order() {
        let mut data: [u32; 20] = core::array::from_fn(|i| i as u32);
        let (train, test) = shuffled_train_test_split(&mut data, 0.25, &mut Rng::new(3));
        assert_eq!((train.len(), test.len()), (15, 5));

        let mut all: Vec<u32> = train.iter().chain(test.iter()).copied().collect();
        assert_ne!(all, (0..20).collect::<Vec<_>>(), "gemischt");
        all.sort_unstable();
        assert_eq!(all, (0..20).collect::<Vec<_>>(), "nichts verloren");
    }

    #[test]
    fn the_split_is_reproducible_with_the_same_seed() {
        let make = |seed| {
            let mut data: [u32; 12] = core::array::from_fn(|i| i as u32);
            let (_, test) = shuffled_train_test_split(&mut data, 0.25, &mut Rng::new(seed));
            test.to_vec()
        };
        assert_eq!(make(5), make(5));
        assert_ne!(make(5), make(6));
    }

    #[test]
    fn a_sorted_data_set_needs_the_shuffle() {
        // 50 Beispiele der Klasse 0, dann 50 der Klasse 1: Ohne Mischen besteht der Testteil nur
        // aus Klasse 1.
        let data: Vec<u8> = (0..100).map(|i| u8::from(i >= 50)).collect();
        let (_, plain_test) = train_test_split(&data, 0.2);
        assert!(plain_test.iter().all(|&c| c == 1));

        let mut shuffled = data.clone();
        let (_, test) = shuffled_train_test_split(&mut shuffled, 0.2, &mut Rng::new(1));
        assert!(test.contains(&0) && test.contains(&1));
    }
}

mod early_stopping {
    use super::*;

    #[test]
    fn it_reports_progress_waiting_and_stop() {
        let mut stopping = EarlyStopping::new(2);
        let verdicts = [0.9, 0.7, 0.8, 0.75].map(|loss| stopping.observe(loss));
        assert_eq!(
            verdicts,
            [
                Progress::Improved,
                Progress::Improved,
                Progress::Waiting,
                Progress::Stop
            ]
        );
        assert_eq!(stopping.best(), Some(0.7));
        assert_eq!(stopping.best_epoch(), 1);
        assert_eq!(stopping.epochs_seen(), 4);
        assert_eq!(stopping.rounds_without_progress(), 2);
    }

    #[test]
    fn an_improvement_resets_the_patience() {
        let mut stopping = EarlyStopping::new(3);
        for loss in [1.0, 1.1, 1.2, 0.5, 0.6, 0.7] {
            assert_ne!(stopping.observe(loss), Progress::Stop);
        }
        assert_eq!(stopping.observe(0.8), Progress::Stop);
        assert_eq!(stopping.best_epoch(), 3);
    }

    #[test]
    fn small_improvements_below_min_delta_do_not_count() {
        let mut stopping = EarlyStopping::new(2).with_min_delta(0.01);
        assert_eq!(stopping.observe(1.0), Progress::Improved);
        assert_eq!(stopping.observe(0.995), Progress::Waiting);
        assert_eq!(stopping.observe(0.99), Progress::Stop);
        assert_eq!(stopping.best(), Some(1.0));
    }

    #[test]
    fn patience_of_one_stops_at_the_first_standstill() {
        let mut stopping = EarlyStopping::new(1);
        assert_eq!(stopping.observe(1.0), Progress::Improved);
        assert_eq!(stopping.observe(1.0), Progress::Stop);
    }

    #[test]
    fn it_works_for_f64() {
        let mut stopping = EarlyStopping::<f64>::new(1);
        assert_eq!(stopping.observe(0.5), Progress::Improved);
    }
}

/// Eine Gerade mit etwas Rauschen: wenig Lerndaten und mehr Parameter als nötig laden zum
/// Überanpassen ein.
fn noisy_line(rng: &mut Rng, n: usize) -> Vec<([f32; 1], [f32; 1])> {
    (0..n)
        .map(|_| {
            let x = rng.uniform(-1.0, 1.0);
            ([x], [2.0 * x + 0.5 + rng.uniform(-0.3, 0.3)])
        })
        .collect()
}

type Wiggly = neuronws_layers::chain::Chain<
    neuronws_layers::chain::Chain<DenseLayer<1, 16>, ActivationLayer<16, Relu>>,
    DenseLayer<16, 1>,
>;

fn wiggly_net(rng: &mut Rng) -> Wiggly {
    DenseLayer::<1, 16>::random(rng, 2.0)
        .then(ActivationLayer::<16, _>::new(Relu))
        .then(DenseLayer::<16, 1>::random(rng, 1.0))
}

mod trainer {
    use super::*;

    #[test]
    fn epoch_shuffled_visits_the_same_examples_in_a_new_order() {
        let mut rng = Rng::new(1);
        let mut data = noisy_line(&mut rng, 12);
        let original = data.clone();
        let mut net = DenseLayer::<1, 1>::new([[0.0]], [0.0]);
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 4);
        trainer.epoch_shuffled(&mut net, &mut data, &mut rng);
        assert_ne!(data, original, "die Reihenfolge hat sich geändert");
        let mut a: Vec<_> = data.iter().map(|(x, _)| x[0].to_bits()).collect();
        let mut b: Vec<_> = original.iter().map(|(x, _)| x[0].to_bits()).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "nichts hinzugekommen, nichts verloren");
    }

    #[test]
    fn fit_shuffled_learns() {
        let mut rng = Rng::new(2);
        let mut data = noisy_line(&mut rng, 40);
        let mut net = DenseLayer::<1, 1>::new([[0.0]], [0.0]);
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 8);
        let before = trainer.evaluate(&net, &data);
        let after = trainer.fit_shuffled(&mut net, &mut data, 80, &mut rng);
        assert!(after < before * 0.1, "{before} -> {after}");
        let [[w]] = *net.weights();
        assert!((w - 2.0).abs() < 0.2);
    }

    #[test]
    fn shuffling_is_reproducible_through_the_seed() {
        let run = |seed| {
            let mut rng = Rng::new(seed);
            let mut data = noisy_line(&mut Rng::new(9), 20);
            let mut net = DenseLayer::<1, 1>::new([[0.0]], [0.0]);
            let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 4);
            trainer.fit_shuffled(&mut net, &mut data, 5, &mut rng);
            net.parameters()
        };
        assert_eq!(run(1), run(1));
        assert_ne!(run(1), run(2));
    }

    #[test]
    fn fit_adaptive_lowers_the_rate_on_a_plateau() {
        // Ein Netz, das nichts mehr lernen kann: konstante Daten, Verlust bleibt gleich.
        let mut net = DenseLayer::<1, 1>::new([[0.0]], [0.0]);
        let data = [([1.0], [0.0]), ([2.0], [0.0])];
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.0), 2);
        let mut plateau = ReduceOnPlateau::new(0.5, 2).with_min_rate(0.0);
        trainer.optimizer_mut().set_learning_rate(0.0);
        // Lernrate 0: der Verlust ändert sich nie, ab der dritten Runde ohne Fortschritt halbiert
        // der Plan. Weil 0 · 0.5 = 0 ist, prüfen wir die Wirkung mit einer Start-Lernrate > 0.
        trainer.optimizer_mut().set_learning_rate(1e-9);
        trainer.fit_adaptive(&mut net, &data, 12, &mut plateau);
        let rate = trainer.optimizer().learning_rate();
        assert!(rate < 1e-9 * 0.5, "{rate}");
        assert!(rate >= 1e-9 / 16.0, "{rate}");
    }

    #[test]
    fn fit_adaptive_keeps_the_rate_while_learning_goes_well() {
        let mut net = DenseLayer::<1, 1>::new([[0.0]], [0.0]);
        let data = [([1.0], [2.0]), ([2.0], [4.0]), ([3.0], [6.0])];
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.01), 3);
        let mut plateau = ReduceOnPlateau::new(0.5, 5);
        trainer.fit_adaptive(&mut net, &data, 20, &mut plateau);
        assert_eq!(trainer.optimizer().learning_rate(), 0.01);
    }

    #[test]
    fn fit_validated_stops_when_the_validation_loss_rises_and_restores_the_best_net() {
        let mut rng = Rng::new(7);
        let train = noisy_line(&mut rng, 8);
        let validation = noisy_line(&mut rng, 40);

        let mut net = wiggly_net(&mut rng);
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.02), 4);
        let report = trainer.fit_validated(
            &mut net,
            &train,
            &validation,
            2000,
            &mut EarlyStopping::new(25),
        );

        assert!(report.stopped_early, "{report:?}");
        assert!(report.epochs < 2000);
        assert_eq!(report.epochs - report.best_epoch - 1, 25);
        // Das Modell ist wieder das der besten Runde: Sein Prüfverlust ist genau der beste.
        let now = trainer.evaluate(&net, &validation);
        assert_eq!(now, report.best_validation_loss);
    }

    #[test]
    fn the_restored_net_beats_the_net_after_the_last_round() {
        let mut rng = Rng::new(7);
        let train = noisy_line(&mut rng, 8);
        let validation = noisy_line(&mut rng, 60);

        // Dieselbe Ausgangslage zweimal: einmal mit Early Stopping, einmal bis zum Ende gelernt.
        let (mut early, mut late) = (wiggly_net(&mut Rng::new(3)), wiggly_net(&mut Rng::new(3)));
        let mut trainer_a = Trainer::new(MeanSquaredError, Sgd::new(0.02), 4);
        let mut trainer_b = Trainer::new(MeanSquaredError, Sgd::new(0.02), 4);
        let report = trainer_a.fit_validated(
            &mut early,
            &train,
            &validation,
            1500,
            &mut EarlyStopping::new(1500),
        );
        trainer_b.fit(&mut late, &train, 1500);

        // Early Stopping (hier ohne vorzeitigen Abbruch, nur mit Wiederherstellung) ist nie
        // schlechter als das Netz am Ende.
        let restored = trainer_a.evaluate(&early, &validation);
        let last = trainer_b.evaluate(&late, &validation);
        assert!(restored <= last + 1e-6, "{restored} gegen {last}");
        assert!(report.best_epoch < 1500);
    }

    #[test]
    fn without_a_standstill_it_runs_all_rounds() {
        let train = [([1.0], [2.0]), ([2.0], [4.0]), ([3.0], [6.0])];
        let validation = [([4.0], [8.0])];
        let mut net = DenseLayer::<1, 1>::new([[0.0]], [0.0]);
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.02), 3);
        let report = trainer.fit_validated(
            &mut net,
            &train,
            &validation,
            30,
            &mut EarlyStopping::new(1000),
        );
        assert_eq!(report.epochs, 30);
        assert!(!report.stopped_early);
        assert_eq!(
            report.best_epoch, 29,
            "der Verlust sinkt bis zur letzten Runde"
        );
        assert!(report.last_train_loss > 0.0);
    }

    #[test]
    fn fit_validated_shuffled_mixes_and_still_restores_the_best_net() {
        let mut rng = Rng::new(11);
        let mut train = noisy_line(&mut rng, 30);
        let validation = noisy_line(&mut rng, 30);
        let mut net = DenseLayer::<1, 1>::new([[0.0]], [0.0]);
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 6);
        let report = trainer.fit_validated_shuffled(
            &mut net,
            &mut train,
            &validation,
            300,
            &mut EarlyStopping::new(15).with_min_delta(1e-4),
            &mut rng,
        );
        assert!(report.stopped_early);
        assert_eq!(
            trainer.evaluate(&net, &validation),
            report.best_validation_loss
        );
        let [[w]] = *net.weights();
        assert!((w - 2.0).abs() < 0.3, "{w}");
    }

    #[test]
    #[should_panic(expected = "max_epochs muss größer als 0 sein")]
    fn zero_rounds_are_refused() {
        let data = [([1.0], [1.0])];
        let mut net = DenseLayer::<1, 1>::new([[0.0]], [0.0]);
        Trainer::new(MeanSquaredError, Sgd::new(0.1), 1).fit_validated(
            &mut net,
            &data,
            &data,
            0,
            &mut EarlyStopping::new(1),
        );
    }

    #[test]
    fn a_report_can_be_compared_and_printed() {
        let train = [([1.0], [2.0])];
        let mut net = DenseLayer::<1, 1>::new([[0.0]], [0.0]);
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.1), 1);
        let report =
            trainer.fit_validated(&mut net, &train, &train, 3, &mut EarlyStopping::new(10));
        assert_eq!(report, report);
        assert!(format!("{report:?}").starts_with("FitReport"));
    }
}
