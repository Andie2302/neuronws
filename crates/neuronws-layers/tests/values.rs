use neuronws_layers::values::Values;

fn collect<V: Values<f32>>(values: &V) -> Vec<f32> {
    (0..V::LEN).map(|i| values.value(i)).collect()
}

#[test]
fn lengths_add_up() {
    assert_eq!(<() as Values<f32>>::LEN, 0);
    assert_eq!(<[f32; 5] as Values<f32>>::LEN, 5);
    assert_eq!(<[[f32; 3]; 2] as Values<f32>>::LEN, 6);
    assert_eq!(<([[f32; 3]; 2], [f32; 4]) as Values<f32>>::LEN, 10);
    assert_eq!(<(([f32; 2], [f32; 3]), [f32; 1]) as Values<f32>>::LEN, 6);
}

#[test]
fn matrices_are_numbered_row_by_row() {
    let matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(collect(&matrix), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn pairs_number_the_left_part_first() {
    let pair = ([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0]);
    assert_eq!(collect(&pair), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn nested_pairs_keep_the_order() {
    let nested = (([1.0], [2.0, 3.0]), [4.0]);
    assert_eq!(collect(&nested), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn value_mut_changes_the_right_place() {
    let mut pair = ([[0.0; 2]; 2], [0.0; 2]);
    *Values::<f32>::value_mut(&mut pair, 3) = 7.0; // letzte Zahl der Matrix
    *Values::<f32>::value_mut(&mut pair, 4) = 9.0; // erste Zahl des Bias
    assert_eq!(pair, ([[0.0, 0.0], [0.0, 7.0]], [9.0, 0.0]));
}

#[test]
fn zeros_has_the_right_shape() {
    let zeros: ([[f32; 2]; 3], [f32; 3]) = Values::<f32>::zeros();
    assert_eq!(zeros, ([[0.0; 2]; 3], [0.0; 3]));
    let nothing: () = Values::<f32>::zeros();
    assert_eq!(nothing, ());
}

#[test]
#[should_panic(expected = "außerhalb")]
fn an_index_out_of_range_panics_for_matrices() {
    let matrix = [[0.0_f32; 2]; 2];
    let _ = Values::<f32>::value(&matrix, 4);
}

#[test]
#[should_panic(expected = "außerhalb")]
fn an_index_out_of_range_panics_for_the_empty_set() {
    let _ = Values::<f32>::value(&(), 0);
}

#[test]
fn it_works_for_f64() {
    let pair = ([1.0_f64, 2.0], [3.0_f64]);
    assert_eq!(Values::<f64>::value(&pair, 2), 3.0);
}
