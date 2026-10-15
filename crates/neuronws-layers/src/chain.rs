use crate::traits::{Data, ForwardData, ForwardPath, Layer};

/// Hintereinanderschaltung zweier Schichten: die Ausgabe von `first` ist die Eingabe von `second`.
///
/// Passen die Typen nicht zusammen, kompiliert es nicht:
///
/// ```compile_fail
/// use neuronws_layers::activation::Relu;
/// use neuronws_layers::activation_layer::ActivationLayer;
/// use neuronws_layers::chain::Chain;
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// // Dense liefert 3 Werte, die ActivationLayer erwartet aber 2.
/// let dense = DenseLayer::new([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [0.0; 3]);
/// let relu = ActivationLayer::<2, _>::new(Relu);
/// let net = Chain::new(dense, relu);
/// net.forward(&[1.0, 2.0]);
/// ```
pub struct Chain<A, B> {
    first: A,
    second: B,
}

impl<A, B> Chain<A, B> {
    pub const fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: Layer, B: Layer> Layer for Chain<A, B> {}

impl<A: Data, B: Data> Data for Chain<A, B> {}

impl<A, B> ForwardData for Chain<A, B>
where
    A: ForwardData,
    B: ForwardData<ForwardInput = A::ForwardOutput>,
{
    type ForwardInput = A::ForwardInput;
    type ForwardOutput = B::ForwardOutput;
}

impl<A, B> ForwardPath for Chain<A, B>
where
    A: ForwardPath,
    B: ForwardPath<ForwardInput = A::ForwardOutput>,
{
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        self.second.forward(&self.first.forward(input))
    }
}

/// Erweiterung für jeden `Layer`: `a.then(b)` ist Kurzform für `Chain::new(a, b)`.
///
/// ```
/// use neuronws_layers::activation::Relu;
/// use neuronws_layers::activation_layer::ActivationLayer;
/// use neuronws_layers::chain::LayerExt;
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::traits::ForwardPath;
///
/// let net = DenseLayer::new([[1.0, -1.0]], [0.0]).then(ActivationLayer::<1, _>::new(Relu));
/// assert_eq!(net.forward(&[1.0, 3.0]), [0.0]);
/// ```
pub trait LayerExt: Layer + Sized {
    fn then<B>(self, next: B) -> Chain<Self, B> {
        Chain::new(self, next)
    }
}

impl<T: Layer> LayerExt for T {}
