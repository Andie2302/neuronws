use crate::scalar::Scalar;
use crate::values::Values;

pub trait Layer {}

pub trait ForwardData {
    type ForwardInput;
    type ForwardOutput;
}
pub trait BackwardData {
    type BackwardInput;
    type BackwardOutput;
}

/// Vorwärtsrechnen ohne Gedächtnis: der **Einsatz** eines fertigen Netzes (Inferenz).
///
/// Es ändert sich nichts am Netz, deshalb reicht `&self`. Zufall wie bei Dropout ist hier aus.
pub trait ForwardPath: ForwardData {
    fn forward(&self, input: &Self::ForwardInput) -> Self::ForwardOutput;
}

/// Vorwärtsrechnen **zum Lernen**: Das Netz merkt sich, was `backward_mut` später braucht
/// (Eingaben, Ausgaben, bei Dropout die Maske).
///
/// Bei Schichten mit Zufall oder laufender Statistik unterscheidet sich das Ergebnis vom
/// [`ForwardPath`]: Dropout lässt hier Werte fallen, beim Einsatz nie.
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
/// Der Besucher (`visitor`) bekommt jeden Parameter und seinen aufsummierten Gradienten, beide
/// veränderbar. Was damit passiert, entscheidet der Optimierer, nicht die Schicht: Ein
/// Standard-Optimierer zieht den Gradienten vom Parameter ab, Gradient-Clipping oder
/// L2-Regularisierung ändern den Gradienten vorher.
pub trait Trainable {
    type Scalar: Scalar;

    /// Ein Platz pro Parameter, in der Reihenfolge von `visit_parameters`.
    ///
    /// Optimierer mit Gedächtnis (Momentum, Adam, …) legen darin pro Parameter ihre Zusatzwerte
    /// ab. Weil der Typ aus dem Modell folgt, kann die Größe nie falsch sein.
    type Shape: Values<Self::Scalar>;

    /// Anzahl der lernbaren Parameter, so viele ruft `visit_parameters` auf.
    ///
    /// Sie folgt aus `Shape` und kann deshalb nie davon abweichen.
    const PARAMETER_COUNT: usize = <Self::Shape as Values<Self::Scalar>>::LEN;

    /// Besucht jeden Parameter mit seinem Gradienten, beide dürfen verändert werden.
    fn visit_parameters(&mut self, visitor: &mut impl FnMut(&mut Self::Scalar, &mut Self::Scalar));

    /// Wie `visit_parameters`, aber nur lesend: Parameter und Gradient.
    fn inspect_parameters(&self, visitor: &mut impl FnMut(Self::Scalar, Self::Scalar));

    /// Setzt alle aufsummierten Gradienten auf 0 zurück.
    fn zero_grad(&mut self);

    /// Kopiert alle Parameter, in der Reihenfolge von `visit_parameters`, in eine Gruppe von der
    /// Form `Shape`.
    ///
    /// Damit lässt sich der Stand eines Netzes festhalten (zum Beispiel der beste bei Early
    /// Stopping) und mit [`set_parameters`](Self::set_parameters) zurückspielen, ganz ohne
    /// `alloc`.
    fn parameters(&self) -> Self::Shape {
        let mut shape = <Self::Shape as Values<Self::Scalar>>::zeros();
        let mut index = 0;
        self.inspect_parameters(&mut |parameter, _| {
            *shape.value_mut(index) = parameter;
            index += 1;
        });
        shape
    }

    /// Gegenstück zu [`parameters`](Self::parameters): schreibt die Parameter zurück.
    /// Gradienten und Zwischenspeicher bleiben unberührt.
    fn set_parameters(&mut self, parameters: &Self::Shape) {
        let mut index = 0;
        self.visit_parameters(&mut |parameter, _| {
            *parameter = parameters.value(index);
            index += 1;
        });
    }
}
