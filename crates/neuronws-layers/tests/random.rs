use neuronws_layers::random::Rng;

#[test]
fn same_seed_gives_same_sequence() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    for _ in 0..100 {
        assert_eq!(a.next_u32(), b.next_u32());
    }
}

#[test]
fn different_seeds_give_different_sequences() {
    let mut a = Rng::new(1);
    let mut b = Rng::new(2);
    let a: [u32; 8] = core::array::from_fn(|_| a.next_u32());
    let b: [u32; 8] = core::array::from_fn(|_| b.next_u32());
    assert_ne!(a, b);
}

#[test]
fn seed_zero_still_produces_values() {
    let mut rng = Rng::new(0);
    let values: [u32; 8] = core::array::from_fn(|_| rng.next_u32());
    assert!(values.iter().any(|&v| v != 0));
    assert!(values.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn next_f32_stays_in_unit_interval_with_mean_near_one_half() {
    let mut rng = Rng::new(7);
    let mut sum = 0.0;
    for _ in 0..10_000 {
        let x = rng.next_f32();
        assert!((0.0..1.0).contains(&x));
        sum += x;
    }
    let mean = sum / 10_000.0;
    assert!((mean - 0.5).abs() < 0.02, "Mittelwert {mean}");
}

#[test]
fn uniform_respects_its_bounds() {
    let mut rng = Rng::new(3);
    for _ in 0..1000 {
        let x = rng.uniform(-2.0, 3.0);
        assert!((-2.0..3.0).contains(&x), "{x}");
    }
}
