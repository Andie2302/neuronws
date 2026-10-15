//! Die eingebauten Datensätze: gleiche Daten für Beispiele und Tests.

use neuronws_layers::datasets::{BLOB_CENTERS, LINE_CLASSES, blobs, line_sample, lines, logic};
use neuronws_layers::metrics::argmax;
use neuronws_layers::random::Rng;

#[test]
fn blobs_have_the_requested_size_and_balanced_classes() {
    let data = blobs::<90>(&mut Rng::new(1));
    assert_eq!(data.len(), 90);
    let mut per_class = [0; 3];
    for (_, target) in &data {
        assert_eq!(target.iter().sum::<f32>(), 1.0, "One-Hot");
        per_class[argmax(target)] += 1;
    }
    assert_eq!(per_class, [30, 30, 30]);
}

#[test]
fn blobs_are_shuffled_and_scattered_around_their_centers() {
    let data = blobs::<300>(&mut Rng::new(2));
    // nicht nach Klassen sortiert
    let first_classes: Vec<usize> = data.iter().take(30).map(|(_, t)| argmax(t)).collect();
    assert!(first_classes.contains(&0) && first_classes.contains(&1) && first_classes.contains(&2));

    for (class, center) in BLOB_CENTERS.iter().enumerate() {
        let points: Vec<[f32; 2]> = data
            .iter()
            .filter(|(_, t)| argmax(t) == class)
            .map(|(p, _)| *p)
            .collect();
        let n = points.len() as f32;
        let mean = [
            points.iter().map(|p| p[0]).sum::<f32>() / n,
            points.iter().map(|p| p[1]).sum::<f32>() / n,
        ];
        assert!((mean[0] - center[0]).abs() < 0.3, "{class}: {mean:?}");
        assert!((mean[1] - center[1]).abs() < 0.3, "{class}: {mean:?}");
        assert!(
            points
                .iter()
                .all(|p| (p[0] - center[0]).abs() <= 1.5 && (p[1] - center[1]).abs() <= 1.5)
        );
    }
}

#[test]
fn blobs_are_deterministic() {
    assert_eq!(blobs::<30>(&mut Rng::new(7)), blobs::<30>(&mut Rng::new(7)));
    assert_ne!(blobs::<30>(&mut Rng::new(7)), blobs::<30>(&mut Rng::new(8)));
}

#[test]
#[should_panic(expected = "N muss durch 3 teilbar sein")]
fn blobs_need_a_multiple_of_three() {
    let _ = blobs::<10>(&mut Rng::new(1));
}

#[test]
fn lines_come_in_three_balanced_classes() {
    let data = lines::<60>(&mut Rng::new(3));
    let mut per_class = [0; 3];
    for (_, target) in &data {
        per_class[argmax(target)] += 1;
    }
    assert_eq!(per_class, [20, 20, 20]);
    assert_eq!(LINE_CLASSES, ["senkrecht", "waagrecht", "diagonal"]);
}

#[test]
fn every_line_image_has_a_bright_line_and_a_dark_background() {
    for class in 0..3 {
        let (image, target) = line_sample(&mut Rng::new(class as u64), class);
        assert_eq!(argmax(&target), class);
        let bright = image[0].iter().flatten().filter(|&&v| v > 0.7).count();
        let dark = image[0]
            .iter()
            .flatten()
            .filter(|&&v| v.abs() < 0.3)
            .count();
        assert_eq!(bright, 6, "eine Linie aus 6 Pixeln");
        assert_eq!(dark, 30);
    }
}

#[test]
fn the_vertical_class_is_really_vertical() {
    let (image, _) = line_sample(&mut Rng::new(5), 0);
    let column: Vec<usize> = (0..6)
        .filter(|&c| (0..6).all(|r| image[0][r][c] > 0.7))
        .collect();
    assert_eq!(column.len(), 1);
    let (horizontal, _) = line_sample(&mut Rng::new(5), 1);
    assert!((0..6).any(|r| (0..6).all(|c| horizontal[0][r][c] > 0.7)));
}

#[test]
fn lines_are_deterministic_and_use_the_random_stream_in_a_fixed_order() {
    assert_eq!(lines::<12>(&mut Rng::new(7)), lines::<12>(&mut Rng::new(7)));
    // Zwei Aufrufe hintereinander setzen den Zufall fort, sie wiederholen sich nicht.
    let mut rng = Rng::new(7);
    let first = lines::<6>(&mut rng);
    let second = lines::<6>(&mut rng);
    assert_ne!(first, second);
}

#[test]
#[should_panic(expected = "es gibt nur 3 Klassen")]
fn line_sample_refuses_an_unknown_class() {
    let _ = line_sample(&mut Rng::new(1), 3);
}

#[test]
fn nand_is_the_opposite_of_and() {
    for (([_, _], and), ([_, _], nand)) in logic::AND.into_iter().zip(logic::NAND) {
        assert_eq!(nand, 1.0 - and);
    }
}

#[test]
fn the_logic_gates_have_the_right_truth_tables() {
    for ([a, b], target) in logic::AND {
        assert_eq!(target, f32::from(a == 1.0 && b == 1.0));
    }
    for ([a, b], target) in logic::OR {
        assert_eq!(target, f32::from(a == 1.0 || b == 1.0));
    }
    for ([a, b], target) in logic::XOR {
        assert_eq!(target, f32::from((a == 1.0) != (b == 1.0)));
    }
}

#[test]
fn a_gate_table_can_be_turned_into_vector_targets() {
    const VECTORS: [([f32; 2], [f32; 1]); 4] = logic::as_vectors(&logic::XOR);
    assert_eq!(VECTORS[1], ([0.0, 1.0], [1.0]));
    assert_eq!(VECTORS[3], ([1.0, 1.0], [0.0]));
}
