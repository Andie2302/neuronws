/// Eine skalare Aktivierungsfunktion `φ` samt Ableitung `φ'`.
///
/// Beides steht in einem Trait, damit man die Ableitung nicht vergessen kann.
/// Beide Methoden bekommen die Eingabe `z` vor der Aktivierung.
pub trait Activation {
    fn apply(&self, z: f32) -> f32;
    fn derivative(&self, z: f32) -> f32;
}

/// φ(z) = z
pub struct Identity;

impl Activation for Identity {
    fn apply(&self, z: f32) -> f32 {
        z
    }

    fn derivative(&self, _z: f32) -> f32 {
        1.0
    }
}

/// Schwellenwert (Heaviside): 1 für z ≥ 0, sonst 0. Die Aktivierung des klassischen Perceptrons.
///
/// Die Ableitung ist überall 0 (bei 0 nicht definiert), deshalb eignet sich `Step` nicht für
/// gradientenbasiertes Lernen. Das Perceptron lernt mit seiner eigenen Regel.
pub struct Step;

impl Activation for Step {
    fn apply(&self, z: f32) -> f32 {
        if z >= 0.0 { 1.0 } else { 0.0 }
    }

    fn derivative(&self, _z: f32) -> f32 {
        0.0
    }
}

/// φ(z) = max(0, z)
pub struct Relu;

impl Activation for Relu {
    fn apply(&self, z: f32) -> f32 {
        z.max(0.0)
    }

    fn derivative(&self, z: f32) -> f32 {
        if z > 0.0 { 1.0 } else { 0.0 }
    }
}
