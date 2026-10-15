//! Verzweigungen: [`Residual`], [`Fork`] und [`Add`].
//!
//! Eine [`Chain`](crate::chain::Chain) kann nur geradeaus. Für Netze mit Abkürzungen (ResNet),
//! Mehrfach-Ausgängen (Encoder eines VAE) oder Zusammenführungen braucht es Bausteine, bei denen
//! sich der Weg teilt oder wieder trifft. Der Kern (`Operation`, `Block`, `Trainable`) bleibt
//! dabei unberührt. Die Bausteine kombinieren fertige Schichten:
//!
//! | Baustein | Eingang | Ausgang | Rückwärts |
//! |---|---|---|---|
//! | [`Residual`]`(f)` | `x` | `x + f(x)` | `g + f.backward(g)` |
//! | [`Fork`]`(a, b)` | `x` | `(a(x), b(x))` | `a.backward(ga) + b.backward(gb)` |
//! | [`Add`] | `(u, v)` | `u + v` | `(g, g)` |
//!
//! **Das ist die Kettenregel an einer Verzweigung:** Fließt ein Wert `x` in zwei Wege, ist der
//! Gradient nach `x` die **Summe** der Gradienten beider Wege. Deshalb addieren `Residual` und
//! `Fork` beim Zurückrechnen.
//!
//! `Residual(f)` ist dasselbe wie `Chain(Fork(Identität, f), Add)`, nur ohne den Umweg über das
//! Paar.

use core::marker::PhantomData;

use crate::block::{Block, Operation};
use crate::init::Initializer;
use crate::random::Rng;
use crate::scalar::Scalar;
use crate::traits::{
    BackwardData, BackwardPathMut, ForwardData, ForwardPath, ForwardPathMut, Layer, Trainable,
};
use crate::values::{self, Values};

/// Addiert zwei gleich geformte Eingaben: `y = u + v`. Hat keine Parameter.
///
/// Das Gegenstück zu [`Fork`]: Zwei Wege laufen hier wieder zusammen. Rückwärts geht der Gradient
/// unverändert in beide Wege, denn `∂(u + v)/∂u = ∂(u + v)/∂v = 1`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Add<V: Values<T>, T: Scalar = f32> {
    shape: PhantomData<(V, T)>,
}

impl<V: Values<T>, T: Scalar> Add<V, T> {
    pub const fn new() -> Self {
        Self { shape: PhantomData }
    }
}

impl<V: Values<T>, T: Scalar> Operation for Add<V, T> {
    type Scalar = T;
    type Input = (V, V);
    type Output = V;
    type Params = ();

    fn forward(&self, _params: &(), (u, v): &(V, V)) -> V {
        values::add(u, v)
    }

    fn backward(
        &self,
        _params: &(),
        _grads: &mut (),
        _input: &(V, V),
        _output: &V,
        grad_output: &V,
    ) -> (V, V) {
        (*grad_output, *grad_output)
    }

    fn init_params<I: Initializer>(&self, _initializer: &I, _rng: &mut Rng) {}
}

/// Eine Additionsschicht über die Form `V`.
pub type AddLayer<V, T = f32> = Block<Add<V, T>>;

impl<V: Values<T>, T: Scalar> Block<Add<V, T>> {
    pub fn new() -> Self {
        Self::from_parts(Add::new(), ())
    }
}

impl<V: Values<T>, T: Scalar> Default for Block<Add<V, T>> {
    fn default() -> Self {
        Self::new()
    }
}

/// Residual-Verbindung ("Abkürzung"): `y = x + f(x)`.
///
/// Das Netz `f` muss nur noch lernen, **was zu `x` dazukommt**, nicht `x` selbst nachbauen. Das
/// macht sehr tiefe Netze trainierbar: Beim Zurückrechnen hat der Gradient mit `g + f.backward(g)`
/// immer den direkten Weg `g` durch die Abkürzung und schrumpft nicht, auch wenn `f` ihn dämpft.
///
/// `f` darf jede Schicht oder `Chain` sein, solange Ein- und Ausgabe dieselbe Form haben (sonst
/// ließe sich `x` nicht addieren).
///
/// ```
/// use neuronws_layers::activation::Relu;
/// use neuronws_layers::activation_layer::ActivationLayer;
/// use neuronws_layers::chain::LayerExt;
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::residual::Residual;
/// use neuronws_layers::traits::ForwardPath;
///
/// // f(x) = Dense → ReLU → Dense, und y = x + f(x)
/// let f = DenseLayer::new([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
///     .then(ActivationLayer::<2, _>::new(Relu))
///     .then(DenseLayer::new([[1.0, 1.0], [-1.0, 1.0]], [0.5, 0.0]));
/// let block = Residual::new(f);
///
/// // x = (2, -1): f(x) = Dense2(ReLU(2, -1)) = Dense2(2, 0) = (2.5, -2), also y = (4.5, -3)
/// assert_eq!(block.forward(&[2.0, -1.0]), [4.5, -3.0]);
/// ```
#[derive(Debug, Clone)]
pub struct Residual<L: ForwardData + BackwardData> {
    inner: L,
    /// Das Ergebnis des letzten `forward_mut`, wird als Referenz zurückgegeben.
    output: L::ForwardOutput,
    /// Der Gradient für den Vorgänger, wird als Referenz zurückgegeben.
    grad_input: L::BackwardOutput,
}

impl<L, T, V> Residual<L>
where
    T: Scalar,
    V: Values<T>,
    L: Trainable<Scalar = T>
        + ForwardData<ForwardInput = V, ForwardOutput = V>
        + BackwardData<BackwardInput = V, BackwardOutput = V>,
{
    pub fn new(inner: L) -> Self {
        Self {
            inner,
            output: V::zeros(),
            grad_input: V::zeros(),
        }
    }
}

impl<L: ForwardData + BackwardData> Residual<L> {
    pub const fn inner(&self) -> &L {
        &self.inner
    }

    pub const fn inner_mut(&mut self) -> &mut L {
        &mut self.inner
    }

    pub fn into_inner(self) -> L {
        self.inner
    }
}

impl<L: Layer + ForwardData + BackwardData> Layer for Residual<L> {}

impl<L: ForwardData + BackwardData> ForwardData for Residual<L> {
    type ForwardInput = L::ForwardInput;
    type ForwardOutput = L::ForwardOutput;
}

impl<L, T, V> ForwardPath for Residual<L>
where
    T: Scalar,
    V: Values<T>,
    L: Trainable<Scalar = T>
        + ForwardPath<ForwardInput = V, ForwardOutput = V>
        + BackwardData<BackwardInput = V, BackwardOutput = V>,
{
    fn forward(&self, input: &V) -> V {
        values::add(input, &self.inner.forward(input))
    }
}

impl<L, T, V> ForwardPathMut for Residual<L>
where
    T: Scalar,
    V: Values<T>,
    L: Trainable<Scalar = T>
        + ForwardPathMut<ForwardInput = V, ForwardOutput = V>
        + BackwardData<BackwardInput = V, BackwardOutput = V>,
{
    fn forward_mut(&mut self, input: &V) -> &V {
        let branch = self.inner.forward_mut(input);
        self.output = values::add(input, branch);
        &self.output
    }
}

impl<L: ForwardData + BackwardData> BackwardData for Residual<L> {
    type BackwardInput = L::BackwardInput;
    type BackwardOutput = L::BackwardOutput;
}

impl<L, T, V> BackwardPathMut for Residual<L>
where
    T: Scalar,
    V: Values<T>,
    L: Trainable<Scalar = T>
        + BackwardPathMut<BackwardInput = V, BackwardOutput = V>
        + ForwardData<ForwardInput = V, ForwardOutput = V>,
{
    /// `∂L/∂x = g + f.backward(g)`: der direkte Weg plus der Weg durch `f`.
    fn backward_mut(&mut self, grad_output: &V) -> &V {
        let through_branch = self.inner.backward_mut(grad_output);
        self.grad_input = values::add(grad_output, through_branch);
        &self.grad_input
    }
}

impl<L: Trainable + ForwardData + BackwardData> Trainable for Residual<L> {
    type Scalar = L::Scalar;
    type Shape = L::Shape;

    fn visit_parameters(&mut self, visitor: &mut impl FnMut(&mut Self::Scalar, &mut Self::Scalar)) {
        self.inner.visit_parameters(visitor);
    }

    fn inspect_parameters(&self, visitor: &mut impl FnMut(Self::Scalar, Self::Scalar)) {
        self.inner.inspect_parameters(visitor);
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }
}

/// Verzweigung: Dieselbe Eingabe läuft durch **zwei** Schichten, die Ausgabe ist das Paar ihrer
/// Ergebnisse `(first(x), second(x))`.
///
/// Beim Zurückrechnen bekommt jede Schicht ihren Teil des Gradienten, und die Gradienten nach der
/// Eingabe aus beiden Wegen werden **addiert** (Kettenregel an einer Verzweigung). Beide
/// Schichten müssen dieselbe Eingabe-Form haben, ihre Ausgaben dürfen verschieden sein.
///
/// Das Paar ist wieder eine `Values`-Gruppe und lässt sich in einer `Chain` weiterreichen: an
/// [`Add`] (zusammenführen), an eine `Reparameterize`-Schicht (Encoder eines VAE: Mittelwert
/// und Varianz) oder an eine Operation, die ein Paar als Eingabe nimmt.
///
/// ```
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::residual::Fork;
/// use neuronws_layers::traits::ForwardPath;
///
/// // Zwei Dense-Schichten sehen dieselben zwei Eingaben: eine liefert die Summe, die andere die
/// // Differenz.
/// let fork = Fork::new(
///     DenseLayer::new([[1.0, 1.0]], [0.0]),
///     DenseLayer::new([[1.0, -1.0]], [0.0]),
/// );
/// assert_eq!(fork.forward(&[5.0, 3.0]), ([8.0], [2.0]));
/// ```
#[derive(Debug, Clone)]
pub struct Fork<A: ForwardData + BackwardData, B: ForwardData + BackwardData> {
    first: A,
    second: B,
    /// Das Paar der Ergebnisse des letzten `forward_mut`.
    output: (A::ForwardOutput, B::ForwardOutput),
    /// Die Summe der Gradienten aus beiden Wegen.
    grad_input: A::BackwardOutput,
}

impl<A, B, T> Fork<A, B>
where
    T: Scalar,
    A: Trainable<Scalar = T> + ForwardData + BackwardData,
    B: Trainable<Scalar = T>
        + ForwardData<ForwardInput = A::ForwardInput>
        + BackwardData<BackwardOutput = A::BackwardOutput>,
    A::ForwardOutput: Values<T>,
    B::ForwardOutput: Values<T>,
    A::BackwardOutput: Values<T>,
{
    pub fn new(first: A, second: B) -> Self {
        Self {
            first,
            second,
            output: <(A::ForwardOutput, B::ForwardOutput) as Values<T>>::zeros(),
            grad_input: <A::BackwardOutput as Values<T>>::zeros(),
        }
    }
}

impl<A: ForwardData + BackwardData, B: ForwardData + BackwardData> Fork<A, B> {
    pub const fn first(&self) -> &A {
        &self.first
    }

    pub const fn second(&self) -> &B {
        &self.second
    }

    pub const fn first_mut(&mut self) -> &mut A {
        &mut self.first
    }

    pub const fn second_mut(&mut self) -> &mut B {
        &mut self.second
    }
}

impl<A, B> Layer for Fork<A, B>
where
    A: Layer + ForwardData + BackwardData,
    B: Layer + ForwardData + BackwardData,
{
}

impl<A, B> ForwardData for Fork<A, B>
where
    A: ForwardData + BackwardData,
    B: ForwardData<ForwardInput = A::ForwardInput> + BackwardData,
{
    type ForwardInput = A::ForwardInput;
    type ForwardOutput = (A::ForwardOutput, B::ForwardOutput);
}

impl<A, B> ForwardPath for Fork<A, B>
where
    A: ForwardPath + BackwardData,
    B: ForwardPath<ForwardInput = A::ForwardInput> + BackwardData,
{
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput {
        (self.first.forward(input), self.second.forward(input))
    }
}

impl<A, B, T> ForwardPathMut for Fork<A, B>
where
    T: Scalar,
    A: Trainable<Scalar = T> + ForwardPathMut + BackwardData,
    B: Trainable<Scalar = T> + ForwardPathMut<ForwardInput = A::ForwardInput> + BackwardData,
    A::ForwardOutput: Values<T>,
    B::ForwardOutput: Values<T>,
{
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput {
        let first = *self.first.forward_mut(input);
        let second = *self.second.forward_mut(input);
        self.output = (first, second);
        &self.output
    }
}

impl<A, B> BackwardData for Fork<A, B>
where
    A: ForwardData + BackwardData,
    B: ForwardData + BackwardData<BackwardOutput = A::BackwardOutput>,
{
    type BackwardInput = (A::BackwardInput, B::BackwardInput);
    type BackwardOutput = A::BackwardOutput;
}

impl<A, B, T> BackwardPathMut for Fork<A, B>
where
    T: Scalar,
    A: Trainable<Scalar = T> + BackwardPathMut + ForwardData,
    B: Trainable<Scalar = T> + BackwardPathMut<BackwardOutput = A::BackwardOutput> + ForwardData,
    A::BackwardOutput: Values<T>,
{
    /// `∂L/∂x = first.backward(g₁) + second.backward(g₂)`
    fn backward_mut(&mut self, grad_output: &Self::BackwardInput) -> &Self::BackwardOutput {
        let through_first = self.first.backward_mut(&grad_output.0);
        let through_second = self.second.backward_mut(&grad_output.1);
        self.grad_input = values::add(through_first, through_second);
        &self.grad_input
    }
}

impl<A, B> Trainable for Fork<A, B>
where
    A: Trainable + ForwardData + BackwardData,
    B: Trainable<Scalar = A::Scalar> + ForwardData + BackwardData,
{
    type Scalar = A::Scalar;
    /// Erst die Plätze der ersten, dann die der zweiten Schicht, wie beim Besuchen.
    type Shape = (A::Shape, B::Shape);

    fn visit_parameters(&mut self, visitor: &mut impl FnMut(&mut Self::Scalar, &mut Self::Scalar)) {
        self.first.visit_parameters(visitor);
        self.second.visit_parameters(visitor);
    }

    fn inspect_parameters(&self, visitor: &mut impl FnMut(Self::Scalar, Self::Scalar)) {
        self.first.inspect_parameters(visitor);
        self.second.inspect_parameters(visitor);
    }

    fn zero_grad(&mut self) {
        self.first.zero_grad();
        self.second.zero_grad();
    }
}
