use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::traits::ForwardPath;

#[test]
fn forward_is_matrix_times_vector_plus_bias() {
    // 2 Eingänge -> 3 Neuronen
    let layer =
        DenseLayer::<2, 3, f32>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [0.5, 0.0, -1.0]);
    assert_eq!(layer.forward(&[1.0, 10.0]), [21.5, 43.0, 64.0]);
}

#[test]
fn one_output_behaves_like_a_single_perceptron() {
    let layer = DenseLayer::<2, 1, f32>::new([[2.0, -1.0]], [0.0]);
    assert_eq!(layer.forward(&[3.0, 4.0]), [2.0]);
}
