//! Schichten aus austauschbaren Teilen bauen, eine eigene Schicht definieren und alles trainieren.
//!
//! Start: `cargo run -p neuronws-layers --features libm --example build_your_own_layer`

use core::marker::PhantomData;

use neuronws_layers::activation::{Identity, Tanh};
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::block::{Block, Operation};
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::Linear;
use neuronws_layers::init::{Initializer, Xavier};
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::neurons::neurons;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::scalar::Scalar;
use neuronws_layers::training::train_epoch;
use neuronws_layers::traits::ForwardPath;

// ---------------------------------------------------------------------------------------------
// Eine eigene Schicht: skaliert und verschiebt jede Eingabe einzeln, y[i] = a[i] · x[i] + b[i].
// Mehr als diese zwei Funktionen muss man nicht schreiben. Zwischenspeicher, Gradienten sammeln,
// die Fähigkeiten zum Vorwärts- und Rückwärtsrechnen und das Lernen liefert `Block`.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct ScaleShift<const N: usize, T: Scalar = f32>(PhantomData<T>);

impl<const N: usize, T: Scalar> Operation for ScaleShift<N, T> {
    type Scalar = T;
    type Input = [T; N];
    type Output = [T; N];
    type Params = ([T; N], [T; N]); // (a, b): das wird gelernt

    fn forward(&self, (a, b): &Self::Params, x: &Self::Input) -> Self::Output {
        core::array::from_fn(|i| a[i] * x[i] + b[i])
    }

    fn backward(
        &self,
        (a, _): &Self::Params,
        (grad_a, grad_b): &mut Self::Params,
        x: &Self::Input,
        _y: &Self::Output,
        g: &Self::Output,
    ) -> Self::Input {
        for i in 0..N {
            grad_a[i] += g[i] * x[i];
            grad_b[i] += g[i];
        }
        core::array::from_fn(|i| g[i] * a[i])
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) -> Self::Params {
        ([T::ONE; N], [T::ZERO; N]) // am Anfang ändert die Schicht nichts
    }
}

fn main() {
    let mut rng = Rng::new(7);

    // Drei Schichten hintereinander, jede aus Berechnung, Startwerten und Aktivierung:
    let mut net = neurons(
        Linear::<2, 4>::new(),
        &Xavier,
        Elementwise::new(Tanh),
        &mut rng,
    )
    .then(Block::initialized(
        ScaleShift::<4>::default(),
        &Xavier,
        &mut rng,
    )) // die eigene
    .then(neurons(
        Linear::<4, 1>::new(),
        &Xavier,
        Elementwise::new(Identity),
        &mut rng,
    ));

    let data: [([f32; 2], [f32; 1]); 4] = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];
    let mut sgd = Sgd::new(0.1);

    for epoch in 0..=3000 {
        let loss = train_epoch(&mut net, &mut sgd, &MeanSquaredError, &data, 4);
        if epoch % 500 == 0 {
            println!("Epoche {epoch:4}: Verlust {loss:e}");
        }
    }

    println!("\nXOR mit einem Netz aus Linear, Tanh, ScaleShift (eigene Schicht), Linear:");
    for (x, target) in &data {
        println!("  {x:?} -> {:.4}  (Ziel {})", net.forward(x)[0], target[0]);
    }
}
