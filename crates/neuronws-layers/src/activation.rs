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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_passes_values_through() {
        assert_eq!(Identity.apply(-2.5), -2.5);
        assert_eq!(Identity.derivative(-2.5), 1.0);
    }

    #[test]
    fn step_switches_at_zero() {
        assert_eq!(Step.apply(-0.1), 0.0);
        assert_eq!(Step.apply(0.0), 1.0);
        assert_eq!(Step.apply(3.0), 1.0);
    }

    #[test]
    fn relu_clips_negative_values() {
        assert_eq!(Relu.apply(-2.0), 0.0);
        assert_eq!(Relu.apply(2.0), 2.0);
        assert_eq!(Relu.derivative(-2.0), 0.0);
        assert_eq!(Relu.derivative(2.0), 1.0);
    }
}
