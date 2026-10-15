//! Diagonal-Schicht, Dense für Matrizen (`LinearRows`) und `Each`.

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::{DenseLayer, DenseRowsLayer, Linear, LinearRows};
use neuronws_layers::diagonal::{Diagonal, DiagonalLayer};
use neuronws_layers::each::Each;
use neuronws_layers::gradcheck::assert_gradients_match;
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

mod diagonal {
    use super::*;

    #[test]
    fn it_scales_and_shifts_every_input_on_its_own() {
        let layer = DiagonalLayer::new([2.0, -1.0, 0.5], [0.0, 1.0, 0.0]);
        assert_eq!(layer.forward(&[3.0, 4.0, 8.0]), [6.0, -3.0, 4.0]);
        assert_eq!(<DiagonalLayer<3> as Trainable>::PARAMETER_COUNT, 6);
    }

    #[test]
    fn it_starts_as_the_identity_whatever_the_initializer() {
        let layer = Block::initialized(
            Diagonal::<3>::new(),
            &neuronws_layers::init::Uniform(5.0),
            &mut neuronws_layers::random::Rng::new(1),
        );
        assert_eq!(layer.weights(), &[1.0; 3]);
        assert_eq!(layer.bias(), &[0.0; 3]);
        assert_eq!(layer.forward(&[7.0, 8.0, 9.0]), [7.0, 8.0, 9.0]);
        assert_eq!(
            DiagonalLayer::<2>::identity().forward(&[1.0, 2.0]),
            [1.0, 2.0]
        );
    }

    #[test]
    fn the_gradients_are_correct() {
        assert_gradients_match(
            &Diagonal::<3, f64>::new(),
            ([0.5, -1.0, 2.0], [0.1, 0.2, 0.3]),
            [1.0, -2.0, 0.5],
            1e-6,
            1e-7,
        );
        assert_gradients_match(
            &Diagonal::<2>::new(),
            ([0.5_f32, -1.0], [0.1, 0.2]),
            [1.0, -2.0],
            1e-2,
            2e-3,
        );
    }

    #[test]
    fn it_learns_a_scaling_and_a_shift_per_input() {
        // Ziel: y0 = 3·x0, y1 = −2·x1 + 1
        let mut layer = DiagonalLayer::<2>::identity();
        let data = [-2.0_f32, -1.0, 0.0, 1.0, 2.0].map(|x| ([x, x], [3.0 * x, -2.0 * x + 1.0]));
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 5);
        trainer.fit(&mut layer, &data, 400);

        let [a0, a1] = *layer.weights();
        let [b0, b1] = *layer.bias();
        assert!((a0 - 3.0).abs() < 1e-3 && (a1 + 2.0).abs() < 1e-3);
        assert!(b0.abs() < 1e-3 && (b1 - 1.0).abs() < 1e-3);
    }

    #[test]
    fn the_gradients_are_collected_per_input() {
        let mut layer = DiagonalLayer::new([2.0, 3.0], [0.0, 0.0]);
        layer.forward_mut(&[5.0, 7.0]);
        let grad_input = *layer.backward_mut(&[1.0, 10.0]);
        assert_eq!(grad_input, [2.0, 30.0]);
        assert_eq!(layer.grad_weights(), &[5.0, 70.0]);
        assert_eq!(layer.grad_bias(), &[1.0, 10.0]);
    }
}

mod rows {
    use super::*;

    fn weights() -> [[f64; 3]; 2] {
        [[0.5, -1.0, 0.25], [2.0, 0.3, -0.7]]
    }

    #[test]
    fn every_row_gets_the_same_affine_map() {
        let rows = DenseRowsLayer::<4, 3, 2, f64>::new(weights(), [0.1, -0.2]);
        let single = DenseLayer::<3, 2, f64>::new(weights(), [0.1, -0.2]);
        let input = [
            [1.0, 2.0, 3.0],
            [0.0, -1.0, 0.5],
            [4.0, 4.0, 4.0],
            [-2.0, 0.0, 1.0],
        ];
        let output = rows.forward(&input);
        for r in 0..4 {
            assert_eq!(output[r], single.forward(&input[r]));
        }
    }

    #[test]
    fn the_gradients_are_the_sum_over_all_rows() {
        let mut rows = DenseRowsLayer::<3, 3, 2, f64>::new(weights(), [0.1, -0.2]);
        let mut single = DenseLayer::<3, 2, f64>::new(weights(), [0.1, -0.2]);
        let input = [[1.0, 2.0, 3.0], [0.0, -1.0, 0.5], [4.0, 4.0, 4.0]];
        let grad = [[1.0, -1.0], [0.5, 2.0], [0.0, 3.0]];

        rows.forward_mut(&input);
        let grad_input = *rows.backward_mut(&grad);

        for r in 0..3 {
            single.forward_mut(&input[r]);
            // Der Gradient für die Eingabe einer Zeile ist der des einzelnen Dense.
            assert_eq!(grad_input[r], *single.backward_mut(&grad[r]));
        }
        assert_eq!(rows.grad_weights(), single.grad_weights());
        assert_eq!(rows.grad_bias(), single.grad_bias());
    }

    #[test]
    fn the_gradients_are_correct() {
        assert_gradients_match(
            &LinearRows::<3, 3, 2, f64>::new(),
            (weights(), [0.1, -0.2]),
            [[1.0, 2.0, 3.0], [0.0, -1.0, 0.5], [4.0, 4.0, 4.0]],
            1e-6,
            1e-7,
        );
    }

    #[test]
    fn the_parameter_count_does_not_depend_on_the_number_of_rows() {
        assert_eq!(
            <DenseRowsLayer<10, 3, 2> as Trainable>::PARAMETER_COUNT,
            <DenseRowsLayer<1, 3, 2> as Trainable>::PARAMETER_COUNT
        );
        assert_eq!(<DenseRowsLayer<10, 3, 2> as Trainable>::PARAMETER_COUNT, 8);
    }

    #[test]
    fn it_learns_a_map_shared_by_all_rows() {
        // Jede Zeile (a, b) → a + 2b
        let mut layer = DenseRowsLayer::<3, 2, 1>::new([[0.0, 0.0]], [0.0]);
        let data = [
            (
                [[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[1.0], [2.0], [3.0]],
            ),
            (
                [[2.0, 1.0], [-1.0, 0.5], [0.0, -2.0]],
                [[4.0], [0.0], [-4.0]],
            ),
        ];
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 2);
        trainer.fit(&mut layer, &data, 500);
        let [[a, b]] = *layer.weights();
        assert!((a - 1.0).abs() < 1e-2 && (b - 2.0).abs() < 1e-2, "{a} {b}");
    }

    #[test]
    fn it_chains_with_row_wise_activations() {
        let net = DenseRowsLayer::<2, 2, 2>::new([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0])
            .then(Block::<Elementwise<[[f32; 2]; 2], Relu>>::new(Relu));
        assert_eq!(
            net.forward(&[[1.0, 2.0], [-3.0, -4.0]]),
            [[1.0, 0.0], [0.0, 4.0]]
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn xavier_start_values_are_within_the_limit() {
        let mut rng = neuronws_layers::random::Rng::new(3);
        let layer = DenseRowsLayer::<5, 4, 6>::xavier(&mut rng);
        let limit = (6.0_f32 / (4.0 + 6.0)).sqrt();
        assert!(layer.weights().iter().flatten().all(|w| w.abs() <= limit));
        assert_eq!(layer.bias(), &[0.0; 6]);
    }

    #[test]
    fn linear_still_works_the_same() {
        // Die gemeinsamen Hilfsfunktionen dürfen `Linear` nicht verändert haben.
        assert_gradients_match(
            &Linear::<3, 2, f64>::new(),
            (weights(), [0.1, -0.2]),
            [1.5, -0.5, 0.8],
            1e-6,
            1e-7,
        );
    }
}

mod each {
    use super::*;

    type Row = Elementwise<[f64; 3], Relu, f64>;

    #[test]
    fn it_applies_the_same_operation_to_every_part() {
        let each = Block::<Each<Row, 2>>::new(Elementwise::new(Relu));
        assert_eq!(
            each.forward(&[[1.0, -2.0, 3.0], [-1.0, 5.0, -6.0]]),
            [[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]]
        );
    }

    #[test]
    fn the_gradient_goes_back_through_every_part() {
        let mut each = Block::<Each<Row, 2>>::new(Elementwise::new(Relu));
        each.forward_mut(&[[1.0, -2.0, 3.0], [-1.0, 5.0, -6.0]]);
        let grad = *each.backward_mut(&[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
        assert_eq!(grad, [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]]);
    }

    #[test]
    fn it_has_no_parameters() {
        assert_eq!(<Block<Each<Row, 4>> as Trainable>::PARAMETER_COUNT, 0);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn each_softmax_normalises_every_row() {
        use neuronws_layers::softmax_layer::Softmax;

        let op = Each::<Softmax<3, f64>, 2>::new(Softmax::new());
        assert_gradients_match(&op, (), [[0.4, -1.2, 2.0], [0.1, 0.3, -0.5]], 1e-6, 1e-7);

        let layer = Block::<Each<Softmax<3, f64>, 2>>::new(Softmax::new());
        let output = layer.forward(&[[0.4, -1.2, 2.0], [0.1, 0.3, -0.5]]);
        for row in output {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }
}
