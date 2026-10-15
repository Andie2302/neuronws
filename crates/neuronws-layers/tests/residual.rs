//! Verzweigungen: Residual, Fork und Add. Der Gradient an einer Verzweigung ist die Summe der Wege.

use neuronws_layers::activation::{Identity, Relu};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::{Chain, LayerExt};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::gradcheck::{assert_gradients_match, assert_layer_gradients_match};
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::residual::{Add, AddLayer, Fork, Residual};
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};

type Body =
    Chain<Chain<DenseLayer<2, 2, f64>, ActivationLayer<2, Relu, f64>>, DenseLayer<2, 2, f64>>;

fn body() -> Body {
    DenseLayer::new([[0.5, -1.0], [2.0, 0.3]], [0.1, -0.2])
        .then(ActivationLayer::<2, _, f64>::new(Relu))
        .then(DenseLayer::new([[1.0, 0.5], [-0.7, 0.9]], [0.0, 0.3]))
}

mod add {
    use super::*;

    #[test]
    fn it_adds_two_inputs() {
        let layer = AddLayer::<[f64; 3], f64>::new();
        assert_eq!(
            layer.forward(&([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])),
            [11.0, 22.0, 33.0]
        );
    }

    #[test]
    fn the_gradient_goes_unchanged_into_both_branches() {
        let mut layer = AddLayer::<[f64; 2], f64>::new();
        layer.forward_mut(&([1.0, 2.0], [3.0, 4.0]));
        let (gu, gv) = *layer.backward_mut(&[0.5, -2.0]);
        assert_eq!(gu, [0.5, -2.0]);
        assert_eq!(gv, [0.5, -2.0]);
    }

    #[test]
    fn the_gradients_are_correct_and_there_are_no_parameters() {
        assert_gradients_match(
            &Add::<[f64; 3], f64>::new(),
            (),
            ([1.0, 2.0, 3.0], [0.5, -1.0, 4.0]),
            1e-6,
            1e-7,
        );
        assert_eq!(<AddLayer<[f32; 5]> as Trainable>::PARAMETER_COUNT, 0);
    }
}

mod residual {
    use super::*;

    #[test]
    fn the_output_is_the_input_plus_the_body() {
        let body_alone = body();
        let residual = Residual::new(body());
        let x = [2.0, -1.0];
        let f = body_alone.forward(&x);
        let y = residual.forward(&x);
        assert_eq!(y, [x[0] + f[0], x[1] + f[1]]);
    }

    #[test]
    fn the_learning_pass_equals_the_inference_pass() {
        let mut residual = Residual::new(body());
        let x = [0.7, 1.3];
        let inference = residual.forward(&x);
        assert_eq!(*residual.forward_mut(&x), inference);
    }

    #[test]
    fn an_identity_body_doubles_the_input() {
        let residual = Residual::new(ActivationLayer::<3, _, f64>::new(Identity));
        assert_eq!(residual.forward(&[1.0, -2.0, 3.0]), [2.0, -4.0, 6.0]);
    }

    #[test]
    fn the_gradient_is_the_direct_path_plus_the_path_through_the_body() {
        let mut residual = Residual::new(body());
        let mut alone = body();
        let x = [0.7, 1.3];
        let g = [1.0, -2.0];

        residual.forward_mut(&x);
        let grad = *residual.backward_mut(&g);

        alone.forward_mut(&x);
        let through = *alone.backward_mut(&g);
        assert_eq!(grad, [g[0] + through[0], g[1] + through[1]]);
    }

    #[test]
    fn the_parameters_are_the_ones_of_the_body_and_get_their_gradients() {
        assert_eq!(
            <Residual<Body> as Trainable>::PARAMETER_COUNT,
            <Body as Trainable>::PARAMETER_COUNT
        );
        let mut residual = Residual::new(body());
        let mut alone = body();
        residual.forward_mut(&[0.7, 1.3]);
        residual.backward_mut(&[1.0, -2.0]);
        alone.forward_mut(&[0.7, 1.3]);
        alone.backward_mut(&[1.0, -2.0]);

        let (mut a, mut b) = (vec![], vec![]);
        residual.inspect_parameters(&mut |_, g| a.push(g));
        alone.inspect_parameters(&mut |_, g| b.push(g));
        assert_eq!(a, b);
    }

    #[test]
    fn the_gradients_of_the_whole_block_are_correct() {
        let mut residual = Residual::new(body());
        assert_layer_gradients_match(&mut residual, [0.7, 1.3], 1e-6, 1e-6);
    }

    #[test]
    fn residuals_can_be_nested_and_chained() {
        let inner = Residual::new(body());
        let mut outer = Residual::new(inner.then(ActivationLayer::<2, _, f64>::new(Identity)));
        assert_layer_gradients_match(&mut outer, [0.4, -0.9], 1e-6, 1e-6);
    }

    #[test]
    fn the_body_can_be_reached() {
        let mut residual = Residual::new(body());
        let _ = residual.inner();
        let _ = residual.inner_mut();
        let _ = residual.into_inner();
    }

    #[test]
    fn a_residual_network_learns() {
        // y = 2x + 1 ist nah an der Identität: Die Abkürzung trägt schon fast alles.
        let mut net = Residual::new(DenseLayer::<1, 1>::new([[0.0]], [0.0]));
        let data = [-2.0_f32, -1.0, 0.0, 1.0, 2.0].map(|x| ([x], [2.0 * x + 1.0]));
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 5);
        trainer.fit(&mut net, &data, 300);
        // x + (a x + b) = 2x + 1 heißt a = 1, b = 1
        let mut params = vec![];
        net.inspect_parameters(&mut |p, _| params.push(p));
        assert!((params[0] - 1.0).abs() < 1e-3 && (params[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn the_shortcut_keeps_the_gradient_alive_through_a_deep_stack() {
        // Vier Schichten, die den Gradienten jeweils auf ein Zehntel dämpfen. Ohne Abkürzung
        // bleiben davon 0.1⁴ = 1e-4. Mit Abkürzung addiert jede Schicht den direkten Weg dazu:
        // (1 + 0.1)⁴ ≈ 1.46.
        type Damper = DenseLayer<1, 1, f64>;
        let damper = || Damper::new([[0.1]], [0.0]);
        let residual = || Residual::new(damper());

        let mut plain = Chain::new(
            Chain::new(damper(), damper()),
            Chain::new(damper(), damper()),
        );
        let mut shortcut = Chain::new(
            Chain::new(residual(), residual()),
            Chain::new(residual(), residual()),
        );

        plain.forward_mut(&[1.0]);
        shortcut.forward_mut(&[1.0]);
        let plain_grad = plain.backward_mut(&[1.0])[0];
        let shortcut_grad = shortcut.backward_mut(&[1.0])[0];
        assert!((plain_grad - 1e-4).abs() < 1e-12, "{plain_grad}");
        assert!(
            (shortcut_grad - 1.1_f64.powi(4)).abs() < 1e-12,
            "{shortcut_grad}"
        );
    }
}

mod fork {
    use super::*;

    fn fork() -> Fork<DenseLayer<2, 3, f64>, DenseLayer<2, 1, f64>> {
        Fork::new(
            DenseLayer::new([[1.0, 2.0], [0.0, -1.0], [0.5, 0.5]], [0.0, 0.1, 0.2]),
            DenseLayer::new([[3.0, -1.0]], [1.0]),
        )
    }

    #[test]
    fn both_branches_see_the_same_input() {
        let fork = fork();
        let (a, b) = fork.forward(&[2.0, 1.0]);
        assert_eq!(a, [4.0, -0.9, 1.7]);
        assert_eq!(b, [6.0]);
    }

    #[test]
    fn the_learning_pass_gives_the_same_pair() {
        let mut fork = fork();
        let x = [2.0, 1.0];
        let inference = fork.forward(&x);
        assert_eq!(*fork.forward_mut(&x), inference);
    }

    #[test]
    fn the_input_gradient_is_the_sum_over_both_branches() {
        let mut fork = fork();
        let mut first =
            DenseLayer::<2, 3, f64>::new([[1.0, 2.0], [0.0, -1.0], [0.5, 0.5]], [0.0, 0.1, 0.2]);
        let mut second = DenseLayer::<2, 1, f64>::new([[3.0, -1.0]], [1.0]);
        let x = [2.0, 1.0];
        let (ga, gb) = ([1.0, -1.0, 2.0], [0.5]);

        fork.forward_mut(&x);
        let grad = *fork.backward_mut(&(ga, gb));

        first.forward_mut(&x);
        second.forward_mut(&x);
        let (from_first, from_second) = (*first.backward_mut(&ga), *second.backward_mut(&gb));
        assert_eq!(
            grad,
            [
                from_first[0] + from_second[0],
                from_first[1] + from_second[1]
            ]
        );
    }

    #[test]
    fn the_gradients_of_the_whole_fork_are_correct() {
        let mut fork = fork();
        assert_layer_gradients_match(&mut fork, [2.0, 1.0], 1e-6, 1e-6);
    }

    #[test]
    fn the_parameters_are_those_of_both_branches_in_order() {
        assert_eq!(
            <Fork<DenseLayer<2, 3>, DenseLayer<2, 1>> as Trainable>::PARAMETER_COUNT,
            9 + 3
        );
        let mut fork = fork();
        let mut seen = vec![];
        fork.visit_parameters(&mut |p, _| seen.push(*p));
        assert_eq!(
            seen,
            [1.0, 2.0, 0.0, -1.0, 0.5, 0.5, 0.0, 0.1, 0.2, 3.0, -1.0, 1.0]
        );
    }

    #[test]
    fn a_fork_followed_by_add_is_a_residual() {
        // Residual(f) ist Chain(Fork(Identität, f), Add), nur ohne das Paar zwischendurch.
        let mut forked = Chain::new(
            Fork::new(ActivationLayer::<2, _, f64>::new(Identity), body()),
            AddLayer::<[f64; 2], f64>::new(),
        );
        let mut residual = Residual::new(body());
        let x = [0.7, 1.3];

        assert_eq!(forked.forward(&x), residual.forward(&x));
        assert_eq!(*forked.forward_mut(&x), *residual.forward_mut(&x));
        let g = [1.0, -2.0];
        assert_eq!(*forked.backward_mut(&g), *residual.backward_mut(&g));

        let (mut a, mut b) = (vec![], vec![]);
        forked.inspect_parameters(&mut |_, g| a.push(g));
        residual.inspect_parameters(&mut |_, g| b.push(g));
        assert_eq!(a, b);
        assert_layer_gradients_match(&mut forked, x, 1e-6, 1e-6);
    }

    #[test]
    fn a_fork_works_inside_a_chain_and_learns() {
        // Zwei Wege aus derselben Eingabe, am Ende addiert: y = (a₁x₁ + a₂x₂) + (b₁x₁ + b₂x₂)
        let mut net = Chain::new(
            Fork::new(
                DenseLayer::<2, 1>::new([[0.1, 0.1]], [0.0]),
                DenseLayer::<2, 1>::new([[0.2, -0.1]], [0.0]),
            ),
            AddLayer::<[f32; 1]>::new(),
        );
        let data = [
            ([1.0_f32, 0.0], [3.0_f32]),
            ([0.0, 1.0], [-1.0]),
            ([1.0, 1.0], [2.0]),
            ([2.0, -1.0], [7.0]),
        ];
        let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.05), 4);
        let before = trainer.evaluate(&net, &data);
        let after = trainer.fit(&mut net, &data, 400);
        assert!(after < before * 1e-3, "{before} -> {after}");
    }

    #[test]
    fn the_branches_can_be_reached() {
        let mut fork = fork();
        let _ = fork.first();
        let _ = fork.second();
        let _ = fork.first_mut();
        let _ = fork.second_mut();
    }
}
