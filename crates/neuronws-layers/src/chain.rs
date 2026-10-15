use crate::traits::{
    BackwardData, BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer, Trainable,
};

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

impl<A, B> ForwardPathMut for Chain<A, B>
where
    A: ForwardPathMut,
    B: ForwardPathMut<ForwardInput = A::ForwardOutput>,
{
    /// Reicht die Referenz auf den Ausgabepuffer von `first` direkt an `second` weiter, ohne Kopie.
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        let mid = self.first.forward_mut(input);
        self.second.forward_mut(mid)
    }
}

/// Rückwärts ist die Reihenfolge gespiegelt: Der Gradient fließt erst durch `second`, dann durch
/// `first`. Der Gradient bezüglich der Eingabe von `second` ist der bezüglich der Ausgabe von `first`.
impl<A, B> BackwardData for Chain<A, B>
where
    A: BackwardData,
    B: BackwardData<BackwardOutput = A::BackwardInput>,
{
    type BackwardInput = B::BackwardInput;
    type BackwardOutput = A::BackwardOutput;
}

impl<A, B> BackwardPathMut for Chain<A, B>
where
    A: BackwardPathMut,
    B: BackwardPathMut<BackwardOutput = A::BackwardInput>,
{
    fn backward_mut(&mut self, grad_output: &Self::BackwardInput) -> &Self::BackwardOutput {
        let mid = self.second.backward_mut(grad_output);
        self.first.backward_mut(mid)
    }
}

impl<A: Trainable, B: Trainable<Scalar = A::Scalar>> Trainable for Chain<A, B> {
    type Scalar = A::Scalar;

    const PARAMETER_COUNT: usize = A::PARAMETER_COUNT + B::PARAMETER_COUNT;

    fn visit_parameters(&mut self, visitor: &mut impl FnMut(&mut Self::Scalar, Self::Scalar)) {
        self.first.visit_parameters(visitor);
        self.second.visit_parameters(visitor);
    }

    fn zero_grad(&mut self) {
        self.first.zero_grad();
        self.second.zero_grad();
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
