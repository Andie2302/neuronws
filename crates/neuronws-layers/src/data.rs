//! Daten aufteilen: Trainingsdaten und Testdaten trennen.
//!
//! Ein Netz soll auf Daten gut sein, die es **nie gesehen** hat. Misst man nur auf den Lerndaten,
//! belohnt man Auswendiglernen. Deshalb legt man einen Teil der Daten beiseite und lernt nur mit
//! dem Rest. Alle Funktionen hier arbeiten auf Scheiben (`&[X]`), ohne Speicher dazuzuholen.
//!
//! ```
//! use neuronws_layers::data::shuffled_train_test_split;
//! use neuronws_layers::random::Rng;
//!
//! let mut data: [u32; 10] = core::array::from_fn(|i| i as u32);
//! let (train, test) = shuffled_train_test_split(&mut data, 0.2, &mut Rng::new(1));
//! assert_eq!((train.len(), test.len()), (8, 2));
//! ```

use crate::random::Rng;

/// Wie viele von `len` Beispielen in den Testteil gehen, mindestens 1 und höchstens `len − 1`.
fn test_len(len: usize, test_fraction: f32) -> usize {
    assert!(len >= 2, "zum Aufteilen braucht es mindestens 2 Beispiele");
    assert!(
        test_fraction > 0.0 && test_fraction < 1.0,
        "test_fraction muss zwischen 0 und 1 liegen, nicht {test_fraction}"
    );
    ((len as f32 * test_fraction + 0.5) as usize).clamp(1, len - 1)
}

/// Teilt `data` in vorne die Trainings- und hinten die Testdaten, **ohne** die Reihenfolge zu
/// ändern. Der Anteil `test_fraction` (zwischen 0 und 1, ohne die Grenzen) geht in den Testteil,
/// gerundet, aber beide Teile haben mindestens ein Beispiel.
///
/// Das ist nur dann in Ordnung, wenn die Daten schon gemischt sind. Liegen sie sortiert vor (erst
/// alle Nullen, dann alle Einsen), landen im Testteil nur Beispiele der letzten Klasse. Dann ist
/// [`shuffled_train_test_split`] richtig.
///
/// # Panics
/// Wenn `data` weniger als 2 Beispiele hat oder `test_fraction` nicht in `(0, 1)` liegt.
pub fn train_test_split<X>(data: &[X], test_fraction: f32) -> (&[X], &[X]) {
    let test = test_len(data.len(), test_fraction);
    data.split_at(data.len() - test)
}

/// Mischt `data` an Ort und Stelle und teilt es dann wie [`train_test_split`] in Trainings- und
/// Testdaten.
///
/// Die Daten des Aufrufers werden dabei umsortiert. Das braucht keinen zusätzlichen Speicher, und
/// die beiden Teile sind Scheiben auf demselben Feld.
///
/// # Panics
/// Wenn `data` weniger als 2 Beispiele hat oder `test_fraction` nicht in `(0, 1)` liegt.
pub fn shuffled_train_test_split<'a, X>(
    data: &'a mut [X],
    test_fraction: f32,
    rng: &mut Rng,
) -> (&'a mut [X], &'a mut [X]) {
    let test = test_len(data.len(), test_fraction);
    rng.shuffle(data);
    let split = data.len() - test;
    data.split_at_mut(split)
}
