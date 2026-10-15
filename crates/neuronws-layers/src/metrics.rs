//! Kennzahlen für Klassifikation: Trefferquote, `argmax` und Verwechslungsmatrix.
//!
//! Damit muss nicht jedes Beispiel `argmax` selbst schreiben. Alle Funktionen arbeiten mit jeder
//! Form (`Values`) und brauchen weder `std` noch `alloc`.

use crate::scalar::Scalar;
use crate::traits::ForwardPath;
use crate::values::Values;

/// Die Nummer des größten Wertes, bei Gleichstand der erste. Bei einem Klassifikations-Netz ist
/// das die vorhergesagte Klasse.
///
/// Werte, die keine Zahl sind (`NaN`), gewinnen nie. Sind alle `NaN` (oder gibt es keine Werte),
/// ist das Ergebnis 0.
///
/// ```
/// use neuronws_layers::metrics::argmax;
///
/// assert_eq!(argmax(&[0.1_f32, 0.7, 0.2]), 1);
/// assert_eq!(argmax(&[3.0_f32, 3.0, 1.0]), 0); // Gleichstand: der erste
/// // jede Form: bei einer Matrix zählt die Nummer in der Reihenfolge von `Values` (zeilenweise)
/// assert_eq!(argmax(&[[0.0_f32, 1.0], [5.0, 2.0]]), 2);
/// ```
pub fn argmax<T: Scalar, V: Values<T>>(values: &V) -> usize {
    let mut best: Option<usize> = None;
    for index in 0..V::LEN {
        let value = values.value(index);
        // `partial_cmp` mit sich selbst ist nur bei `NaN` leer.
        let is_number = value.partial_cmp(&value).is_some();
        if is_number && best.is_none_or(|best| value > values.value(best)) {
            best = Some(index);
        }
    }
    best.unwrap_or(0)
}

/// Anteil der Beispiele, bei denen die größte Ausgabe an der Stelle der größten Zielzahl steht
/// (zwischen 0 und 1).
///
/// Das Ziel ist wie bei `CategoricalCrossEntropy` ein One-Hot-Vektor: eine 1 bei der richtigen
/// Klasse, sonst 0. Gemessen wird nur mit dem Einsatzweg (`forward`), das Netz ändert sich nicht.
///
/// ```
/// use neuronws_layers::activation::Identity;
/// use neuronws_layers::activation_layer::ActivationLayer;
/// use neuronws_layers::metrics::accuracy;
///
/// // Ein Netz, das seine Eingabe unverändert weitergibt
/// let net = ActivationLayer::<3, _>::new(Identity);
/// let data = [
///     ([0.9, 0.1, 0.0], [1.0, 0.0, 0.0]), // richtig
///     ([0.2, 0.7, 0.1], [0.0, 1.0, 0.0]), // richtig
///     ([0.5, 0.4, 0.1], [0.0, 0.0, 1.0]), // falsch
///     ([0.1, 0.2, 0.7], [0.0, 0.0, 1.0]), // richtig
/// ];
/// assert_eq!(accuracy(&net, &data), 0.75);
/// ```
///
/// # Panics
/// Wenn `data` leer ist.
pub fn accuracy<T, M, I, Out>(model: &M, data: &[(I, Out)]) -> T
where
    T: Scalar,
    Out: Values<T>,
    M: ForwardPath<ForwardInput = I, ForwardOutput = Out>,
{
    assert!(!data.is_empty(), "data darf nicht leer sein");
    let correct = data
        .iter()
        .filter(|(input, target)| argmax(&model.forward(input)) == argmax(target))
        .count();
    T::from_usize(correct) / T::from_usize(data.len())
}

/// Wie [`accuracy`], aber für Ausgaben, die **einzeln** ja oder nein sagen (zum Beispiel XOR mit
/// einem Ausgang): Jeder Ausgabewert zählt als "ja" ab `0,5`, ein Beispiel ist richtig, wenn
/// **alle** seine Ausgaben zum Ziel passen (Ziel ab `0,5` ist "ja").
///
/// # Panics
/// Wenn `data` leer ist.
pub fn binary_accuracy<T, M, I, Out>(model: &M, data: &[(I, Out)]) -> T
where
    T: Scalar,
    Out: Values<T>,
    M: ForwardPath<ForwardInput = I, ForwardOutput = Out>,
{
    assert!(!data.is_empty(), "data darf nicht leer sein");
    let half = T::from_f32(0.5);
    let correct = data
        .iter()
        .filter(|(input, target)| {
            let output = model.forward(input);
            (0..Out::LEN).all(|i| (output.value(i) >= half) == (target.value(i) >= half))
        })
        .count();
    T::from_usize(correct) / T::from_usize(data.len())
}

/// Die Verwechslungsmatrix (Confusion Matrix): `matrix[richtig][vorhergesagt]` zählt, wie oft ein
/// Beispiel der Klasse `richtig` als Klasse `vorhergesagt` erkannt wurde. Auf der Diagonalen
/// stehen die Treffer, alles andere sind Verwechslungen.
///
/// `K` ist die Zahl der Klassen. Sie muss zur Länge der Ausgabe passen.
///
/// ```
/// use neuronws_layers::activation::Identity;
/// use neuronws_layers::activation_layer::ActivationLayer;
/// use neuronws_layers::metrics::confusion_matrix;
///
/// let net = ActivationLayer::<2, _>::new(Identity);
/// let data = [
///     ([0.9, 0.1], [1.0, 0.0]),
///     ([0.6, 0.4], [1.0, 0.0]),
///     ([0.7, 0.3], [0.0, 1.0]), // eine 1 wurde für eine 0 gehalten
///     ([0.2, 0.8], [0.0, 1.0]),
/// ];
/// assert_eq!(confusion_matrix::<2, _, _, _, _>(&net, &data), [[2, 0], [1, 1]]);
/// ```
///
/// # Panics
/// Wenn eine Klasse (aus Ausgabe oder Ziel) nicht kleiner als `K` ist.
pub fn confusion_matrix<const K: usize, T, M, I, Out>(
    model: &M,
    data: &[(I, Out)],
) -> [[usize; K]; K]
where
    T: Scalar,
    Out: Values<T>,
    M: ForwardPath<ForwardInput = I, ForwardOutput = Out>,
{
    let mut matrix = [[0; K]; K];
    for (input, target) in data {
        let (right, predicted) = (argmax(target), argmax(&model.forward(input)));
        assert!(
            right < K && predicted < K,
            "Klasse {right} oder {predicted} passt nicht zu K = {K}"
        );
        matrix[right][predicted] += 1;
    }
    matrix
}
