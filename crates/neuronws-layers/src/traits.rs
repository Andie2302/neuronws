pub trait Layer {}

pub trait ForwardData {
    type ForwardInput;
    type ForwardOutput;
}
pub trait BackwardData {
    type BackwardInput;
    type BackwardOutput;
}

pub trait ForwardPath: ForwardData {
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput;
}

pub trait ForwardPathMut: ForwardData {
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &Self::ForwardOutput;
}

pub trait BackwardPath: BackwardData {
    fn backward(&self, input: &Self::BackwardInput) -> Self::BackwardOutput;
}

pub trait BackwardPathMut: BackwardData {
    fn backward_mut(&mut self, input: &Self::BackwardInput) -> &Self::BackwardOutput;
}

/// Die Fähigkeit, lernbare Parameter samt Gradienten herzugeben.
///
/// Der Besucher (`visitor`) bekommt jeden Parameter und seinen aufsummierten Gradienten.
/// Was damit passiert, entscheidet der Optimierer, nicht die Schicht.
pub trait Trainable {
    type Scalar;

    /// Anzahl der lernbaren Parameter, so viele ruft `visit_parameters` auf.
    const PARAMETER_COUNT: usize;

    fn visit_parameters(&mut self, visitor: &mut impl FnMut(&mut Self::Scalar, Self::Scalar));

    /// Setzt alle aufsummierten Gradienten auf 0 zurück.
    fn zero_grad(&mut self);
}
