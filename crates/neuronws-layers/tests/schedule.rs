use neuronws_layers::schedule::{Constant, LinearDecay, Schedule, StepDecay};

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} gegen {b}");
}

#[test]
fn constant_keeps_the_base_rate() {
    for epoch in [0, 1, 1000] {
        assert_eq!(Constant.rate(epoch, 0.3_f32), 0.3);
    }
}

#[test]
fn step_decay_multiplies_every_n_epochs() {
    let schedule = StepDecay::new(0.5_f64, 10);
    assert_close(schedule.rate(0, 1.0), 1.0);
    assert_close(schedule.rate(9, 1.0), 1.0);
    assert_close(schedule.rate(10, 1.0), 0.5);
    assert_close(schedule.rate(19, 1.0), 0.5);
    assert_close(schedule.rate(20, 1.0), 0.25);
    assert_close(schedule.rate(35, 0.8), 0.8 * 0.125);
}

#[test]
fn step_decay_every_one_is_exponential_decay() {
    let schedule = StepDecay::new(0.9_f64, 1);
    assert_close(schedule.rate(5, 2.0), 2.0 * 0.9_f64.powi(5));
}

#[test]
fn linear_decay_goes_straight_to_the_final_fraction_and_stays() {
    let schedule = LinearDecay::new(0.1_f64, 10);
    assert_close(schedule.rate(0, 1.0), 1.0);
    assert_close(schedule.rate(5, 1.0), 0.55);
    assert_close(schedule.rate(10, 1.0), 0.1);
    assert_close(schedule.rate(500, 1.0), 0.1);
    assert_close(schedule.rate(5, 2.0), 1.1);
}

#[test]
#[should_panic(expected = "every muss größer als 0 sein")]
fn step_decay_rejects_zero_interval() {
    let _ = StepDecay::new(0.5_f32, 0);
}

#[test]
#[should_panic(expected = "epochs muss größer als 0 sein")]
fn linear_decay_rejects_zero_epochs() {
    let _ = LinearDecay::new(0.5_f32, 0);
}

/// Ein eigener Plan ist ein einzelnes `impl`.
#[test]
fn a_custom_schedule_is_one_impl() {
    struct Alternate;
    impl Schedule<f32> for Alternate {
        fn rate(&self, epoch: usize, base: f32) -> f32 {
            if epoch.is_multiple_of(2) {
                base
            } else {
                base / 10.0
            }
        }
    }
    assert_eq!(Alternate.rate(0, 1.0), 1.0);
    assert_eq!(Alternate.rate(1, 1.0), 0.1);
}
