use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::optimizer::{HasLearningRate, Momentum, Sgd};
use neuronws_layers::schedule::{Constant, Schedule, StepDecay};
use neuronws_layers::trainer::Trainer;
use neuronws_layers::training::{mean_loss, train_epoch};

type Line = DenseLayer<1, 1>;

fn data() -> Vec<([f32; 1], [f32; 1])> {
    [-2.0, -1.0, 0.0, 1.0, 2.0]
        .map(|x: f32| ([x], [2.0 * x + 1.0]))
        .to_vec()
}

#[test]
fn an_epoch_is_exactly_train_epoch() {
    let data = data();
    let (mut a, mut b) = (Line::new([[0.0]], [0.0]), Line::new([[0.0]], [0.0]));
    let mut trainer = Trainer::new(MeanSquaredError, Momentum::<Line>::new(0.02, 0.9), 2);
    let mut momentum = Momentum::<Line>::new(0.02, 0.9);

    for _ in 0..10 {
        let by_trainer = trainer.epoch(&mut a, &data);
        let by_hand = train_epoch(&mut b, &mut momentum, &MeanSquaredError, &data, 2);
        assert_eq!(by_trainer, by_hand);
    }
    assert_eq!(a.weights(), b.weights());
    assert_eq!(a.bias(), b.bias());
}

#[test]
fn online_is_batch_size_one() {
    let trainer = Trainer::online(MeanSquaredError, Sgd::new(0.1_f32));
    assert_eq!(trainer.batch_size(), 1);
}

#[test]
fn fit_runs_the_given_number_of_epochs_and_returns_the_last_loss() {
    let data = data();
    let (mut a, mut b) = (Line::new([[0.0]], [0.0]), Line::new([[0.0]], [0.0]));
    let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.02), 5);
    let mut sgd = Sgd::new(0.02);

    let fitted = trainer.fit(&mut a, &data, 7);
    let mut last = 0.0;
    for _ in 0..7 {
        last = train_epoch(&mut b, &mut sgd, &MeanSquaredError, &data, 5);
    }
    assert_eq!(fitted, last);
    assert_eq!(a.weights(), b.weights());
}

#[test]
fn every_batch_size_learns_the_line() {
    let data = data();
    for batch_size in [1, 2, 5, 100] {
        let mut line = Line::new([[0.0]], [0.0]);
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.02), batch_size);
        trainer.fit(&mut line, &data, 1500);
        assert!((line.weights()[0][0] - 2.0).abs() < 1e-2, "{batch_size}");
        assert!((line.bias()[0] - 1.0).abs() < 1e-2, "{batch_size}");
    }
}

#[test]
fn evaluate_does_not_learn() {
    let data = data();
    let line = Line::new([[0.5]], [0.0]);
    let trainer = Trainer::new(MeanSquaredError, Sgd::new(0.1), 5);
    assert_eq!(
        trainer.evaluate(&line, &data),
        mean_loss(&line, &MeanSquaredError, &data)
    );
    assert_eq!(line.weights(), &[[0.5]]);
}

#[test]
fn the_same_trainer_can_train_different_networks() {
    let data = data();
    let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.02), 5);
    let (mut a, mut b) = (Line::new([[0.0]], [0.0]), Line::new([[5.0]], [-5.0]));
    trainer.fit(&mut a, &data, 1500);
    trainer.fit(&mut b, &data, 1500);
    assert!((a.weights()[0][0] - b.weights()[0][0]).abs() < 2e-2);
}

#[test]
fn accessors_expose_the_parts() {
    let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.1_f32), 4);
    assert_eq!(trainer.optimizer().learning_rate(), 0.1);
    trainer.optimizer_mut().set_learning_rate(0.2);
    assert_eq!(trainer.optimizer().learning_rate(), 0.2);
    let _ = trainer.loss();
}

#[test]
#[should_panic(expected = "batch_size muss größer als 0 sein")]
fn zero_batch_size_is_rejected() {
    let _ = Trainer::new(MeanSquaredError, Sgd::new(0.1_f32), 0);
}

#[test]
#[should_panic(expected = "epochs muss größer als 0 sein")]
fn zero_epochs_are_rejected() {
    let mut line = Line::new([[0.0]], [0.0]);
    Trainer::online(MeanSquaredError, Sgd::new(0.1)).fit(&mut line, &data(), 0);
}

#[test]
fn a_schedule_sets_the_learning_rate_before_every_epoch() {
    struct Spy(std::cell::RefCell<Vec<(usize, f32)>>);
    impl Schedule<f32> for Spy {
        fn rate(&self, epoch: usize, base: f32) -> f32 {
            self.0.borrow_mut().push((epoch, base));
            base / (epoch as f32 + 1.0)
        }
    }

    let mut line = Line::new([[0.0]], [0.0]);
    let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.4), 5);
    let spy = Spy(Default::default());
    trainer.fit_scheduled(&mut line, &data(), 4, &spy);

    // Die Start-Lernrate wird einmal gelesen, danach nicht mehr vom Plan überschrieben.
    assert_eq!(*spy.0.borrow(), [(0, 0.4), (1, 0.4), (2, 0.4), (3, 0.4)]);
    assert!((trainer.optimizer().learning_rate() - 0.1).abs() < 1e-6);
}

#[test]
fn a_constant_schedule_is_plain_fit() {
    let data = data();
    let (mut a, mut b) = (Line::new([[0.0]], [0.0]), Line::new([[0.0]], [0.0]));
    Trainer::new(MeanSquaredError, Sgd::new(0.02), 5).fit(&mut a, &data, 30);
    Trainer::new(MeanSquaredError, Sgd::new(0.02), 5).fit_scheduled(&mut b, &data, 30, &Constant);
    assert_eq!(a.weights(), b.weights());
}

#[test]
fn step_decay_ends_with_the_decayed_rate() {
    let mut line = Line::new([[0.0]], [0.0]);
    let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.08_f32), 5);
    trainer.fit_scheduled(&mut line, &data(), 25, &StepDecay::new(0.5, 10));
    // Runden 0..9: 0.08, 10..19: 0.04, 20..24: 0.02
    assert!((trainer.optimizer().learning_rate() - 0.02).abs() < 1e-7);
}
