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

#[test]
fn images_with_channels_are_numbered_channel_then_row_then_column() {
    // 2 Kanäle, 2 Zeilen, 3 Spalten
    let image = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ];
    assert_eq!(<[[[f32; 3]; 2]; 2] as Values<f32>>::LEN, 12);
    assert_eq!(
        collect(&image),
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ]
    );
}

#[test]
fn four_dimensional_shapes_work_too() {
    let mut batch = [[[[0.0_f32; 2]; 3]; 2]; 2];
    assert_eq!(<[[[[f32; 2]; 3]; 2]; 2] as Values<f32>>::LEN, 24);
    for i in 0..24 {
        *batch.value_mut(i) = i as f32;
    }
    assert_eq!(batch[1][0][2][1], 17.0); // 1·12 + 0·6 + 2·2 + 1
    assert_eq!(
        collect(&batch),
        (0..24).map(|i| i as f32).collect::<Vec<_>>()
    );
}

#[test]
fn a_single_number_is_a_group_of_one() {
    assert_eq!(<f32 as Values<f32>>::LEN, 1);
    assert_eq!(collect(&4.5_f32), [4.5]);
    let mut x = 1.0_f32;
    *x.value_mut(0) = 2.0;
    assert_eq!(x, 2.0);
}

#[test]
fn tensors_can_be_part_of_pairs() {
    let pair = ([[[1.0, 2.0]; 2]; 1], [3.0]);
    assert_eq!(collect(&pair), [1.0, 2.0, 1.0, 2.0, 3.0]);
}

#[test]
fn zeros_has_the_right_shape_for_any_depth() {
    let zeros = <[[[f64; 3]; 2]; 4] as Values<f64>>::zeros();
    assert_eq!(zeros, [[[0.0; 3]; 2]; 4]);
}

#[test]
#[should_panic(expected = "außerhalb von 12 Werten")]
fn reading_past_the_end_of_a_tensor_panics() {
    let image = [[[0.0_f32; 3]; 2]; 2];
    image.value(12);
}

#[test]
#[should_panic(expected = "außerhalb von 1 Wert")]
fn a_single_number_has_only_index_zero() {
    1.0_f32.value(1);
}

#[test]
#[should_panic(expected = "außerhalb von 0 Werten")]
fn empty_inner_shapes_have_no_values() {
    // Ein Array mit leerer Innenform: LEN ist 0, der Zugriff muss sauber scheitern (nicht durch 0 teilen).
    let empty = [[0.0_f32; 0]; 3];
    empty.value(0);
}
