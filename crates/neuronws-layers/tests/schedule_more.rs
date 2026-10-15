//! Warmup, Cosinus-Abfall und der Plan, der auf den Verlust reagiert.

use neuronws_layers::schedule::{
    AdaptiveSchedule, Constant, LinearDecay, ReduceOnPlateau, Schedule, StepDecay, Warmup,
};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} gegen {b}");
}

#[test]
fn warmup_ramps_up_linearly_and_then_hands_over() {
    let warmup = Warmup::new(4, Constant);
    let rates: Vec<f64> = (0..7).map(|epoch| warmup.rate(epoch, 0.2)).collect();
    for (rate, expected) in rates.iter().zip([0.05, 0.1, 0.15, 0.2, 0.2, 0.2, 0.2]) {
        close(*rate, expected);
    }
}

#[test]
fn the_inner_schedule_starts_counting_after_the_warmup() {
    // StepDecay halbiert alle 2 Runden. Runde 0 des inneren Plans ist die Runde nach dem Anlauf.
    let warmup = Warmup::new(3, StepDecay::new(0.5, 2));
    let rates: Vec<f64> = (0..8).map(|epoch| warmup.rate(epoch, 1.0)).collect();
    let expected = [1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.25];
    for (rate, expected) in rates.iter().zip(expected) {
        close(*rate, expected);
    }
}

#[test]
fn warmup_of_zero_rounds_is_just_the_inner_schedule() {
    let warmup = Warmup::new(0, LinearDecay::new(0.0, 10));
    for epoch in 0..12 {
        close(
            warmup.rate(epoch, 0.5),
            LinearDecay::new(0.0, 10).rate(epoch, 0.5),
        );
    }
}

#[test]
fn warmup_works_for_f32() {
    let rate: f32 = Warmup::new(2, Constant).rate(0, 0.1);
    assert_eq!(rate, 0.05);
}

#[cfg(any(feature = "std", feature = "libm"))]
mod cosine {
    use super::*;
    use neuronws_layers::schedule::CosineDecay;

    #[test]
    fn it_starts_at_base_and_ends_at_the_final_fraction() {
        let cosine = CosineDecay::new(0.1, 20);
        close(cosine.rate(0, 1.0), 1.0);
        close(cosine.rate(20, 1.0), 0.1);
        // ab dann bleibt es dort
        close(cosine.rate(21, 1.0), 0.1);
        close(cosine.rate(1000, 1.0), 0.1);
    }

    #[test]
    fn the_middle_is_exactly_halfway() {
        let cosine = CosineDecay::new(0.0, 10);
        close(cosine.rate(5, 2.0), 1.0);
        let cosine = CosineDecay::new(0.2, 10);
        close(cosine.rate(5, 1.0), 0.6);
    }

    #[test]
    fn it_never_rises_and_is_flat_at_both_ends() {
        let cosine = CosineDecay::new(0.0, 100);
        let rates: Vec<f64> = (0..=100).map(|e| cosine.rate(e, 1.0)).collect();
        assert!(rates.windows(2).all(|w| w[1] <= w[0] + 1e-15));
        // flach am Anfang und am Ende, steil in der Mitte
        let drop_start = rates[0] - rates[5];
        let drop_middle = rates[47] - rates[52];
        let drop_end = rates[95] - rates[100];
        assert!(drop_middle > 10.0 * drop_start && drop_middle > 10.0 * drop_end);
    }

    #[test]
    fn it_works_for_f32_and_combines_with_warmup() {
        let schedule = Warmup::new(5, CosineDecay::new(0.0_f32, 10));
        assert_eq!(schedule.rate(4, 0.1), 0.1);
        assert_eq!(schedule.rate(5, 0.1), 0.1);
        assert!(schedule.rate(10, 0.1) < 0.1);
        assert!(schedule.rate(15, 0.1).abs() < 1e-7);
    }

    #[test]
    #[should_panic(expected = "epochs muss größer als 0 sein")]
    fn zero_epochs_are_refused() {
        let _ = CosineDecay::new(0.0_f32, 0);
    }
}

#[test]
fn reduce_on_plateau_keeps_the_rate_while_the_loss_improves() {
    let mut plateau = ReduceOnPlateau::new(0.5, 2);
    let mut rate = 0.1_f64;
    for loss in [1.0, 0.9, 0.8, 0.7, 0.6] {
        rate = plateau.next_rate(loss, rate);
        assert_eq!(rate, 0.1);
    }
    assert_eq!(plateau.best(), Some(0.6));
    assert_eq!(plateau.rounds_without_progress(), 0);
}

#[test]
fn reduce_on_plateau_waits_for_the_patience() {
    let mut plateau = ReduceOnPlateau::new(0.5, 3);
    let mut rate = 1.0_f64;
    rate = plateau.next_rate(1.0, rate); // erster Wert zählt als Fortschritt
    let mut history = vec![];
    for _ in 0..4 {
        rate = plateau.next_rate(1.0, rate); // kein Fortschritt mehr
        history.push(rate);
    }
    // Drei Runden Geduld: erst die vierte Runde ohne Fortschritt halbiert.
    assert_eq!(history, [1.0, 1.0, 1.0, 0.5]);
}

#[test]
fn reduce_on_plateau_restarts_the_count_after_a_reduction() {
    let mut plateau = ReduceOnPlateau::new(0.1, 1);
    let mut rate = 1.0_f64;
    rate = plateau.next_rate(1.0, rate);
    let mut rates = vec![];
    for _ in 0..6 {
        rate = plateau.next_rate(1.0, rate);
        rates.push(rate);
    }
    // alle zwei Runden ohne Fortschritt ein Zehntel
    for (got, expected) in rates.iter().zip([1.0, 0.1, 0.1, 0.01, 0.01, 0.001]) {
        close(*got, expected);
    }
}

#[test]
fn an_improvement_resets_the_count() {
    let mut plateau = ReduceOnPlateau::new(0.5, 2);
    let mut rate = 1.0_f64;
    for loss in [1.0, 1.0, 1.0, 0.5, 0.5, 0.5] {
        rate = plateau.next_rate(loss, rate);
    }
    // Zweimal schlechter, dann besser, dann wieder nur zweimal schlechter: nie die Geduld überschritten.
    assert_eq!(rate, 1.0);
}

#[test]
fn small_improvements_below_min_delta_do_not_count() {
    let mut plateau = ReduceOnPlateau::new(0.5, 1).with_min_delta(0.01);
    let mut rate = 1.0_f64;
    for loss in [1.0, 0.999, 0.998, 0.997] {
        rate = plateau.next_rate(loss, rate);
    }
    assert_eq!(rate, 0.5);

    // ohne min_delta zählt jede kleine Verbesserung
    let mut plateau = ReduceOnPlateau::new(0.5, 1);
    let mut rate = 1.0_f64;
    for loss in [1.0, 0.999, 0.998, 0.997] {
        rate = plateau.next_rate(loss, rate);
    }
    assert_eq!(rate, 1.0);
}

#[test]
fn the_rate_never_falls_below_the_minimum() {
    let mut plateau = ReduceOnPlateau::new(0.1, 0).with_min_rate(0.005);
    let mut rate = 1.0_f64;
    rate = plateau.next_rate(1.0, rate);
    for _ in 0..10 {
        rate = plateau.next_rate(1.0, rate);
    }
    assert_eq!(rate, 0.005);
}
