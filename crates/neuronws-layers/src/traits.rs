pub trait Layer {}

pub trait Data {}

pub trait ForwardData: Data {
    type ForwardInput;
    type ForwardOutput;
}
pub trait BackwardData: Data {
    type BackwardInput;
    type BackwardOutput;
}

pub trait ForwardPath: ForwardData {
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput;
}

pub trait ForwardPathMut: ForwardData {
    fn forward_mut(&mut self, input: &Self::ForwardInput) -> &mut Self::ForwardOutput;
}

pub trait BackwardPath: BackwardData {
    fn backward(&self, input: &Self::BackwardInput) -> Self::BackwardOutput;
}

pub trait BackwardPathMut: BackwardData {
    fn backward_mut(&mut self, input: &Self::BackwardInput) -> &mut Self::BackwardOutput;
}
