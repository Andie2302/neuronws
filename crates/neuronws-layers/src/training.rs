//! Hilfsfunktionen, damit die Trainingsschleife nicht in jedem Programm neu geschrieben wird.

use crate::loss::Loss;
use crate::optimizer::Optimizer;
use crate::scalar::Scalar;
use crate::traits::{BackwardPathMut, ForwardPath, ForwardPathMut, Trainable};
use crate::values::Values;

/// Eine Runde über alle Beispiele in Mini-Batches. Liefert den mittleren Verlust der Runde.
///
/// Die Beispiele werden der Reihe nach in Gruppen zu `batch_size` verarbeitet. Nach jeder Gruppe
/// macht der Optimierer einen Schritt. Ist die letzte Gruppe kleiner, wird mit ihrer wirklichen
/// Größe gemittelt. `batch_size == 1` ist klassisches SGD, `batch_size == data.len()` ein
/// Schritt pro Runde. Für eine andere Reihenfolge pro Runde vorher `Rng::shuffle` aufrufen.
///
/// Der gemeldete Verlust ist der **vor** dem jeweiligen Schritt gemessene.
///
/// # Panics
/// Wenn `batch_size` 0 ist oder `data` leer.
pub fn train_epoch<M, O, L, T, I, Out>(
    model: &mut M,
    optimizer: &mut O,
    loss: &L,
    data: &[(I, Out)],
    batch_size: usize,
) -> T
where
    T: Scalar,
    Out: Values<T>,
    M: ForwardPathMut<ForwardInput = I, ForwardOutput = Out>
        + BackwardPathMut<BackwardInput = Out>
        + Trainable<Scalar = T>,
    O: Optimizer<M>,
    L: Loss,
{
    assert!(batch_size > 0, "batch_size muss größer als 0 sein");
    assert!(!data.is_empty(), "data darf nicht leer sein");

    let mut total = T::ZERO;
    for batch in data.chunks(batch_size) {
        for (input, target) in batch {
            let output = *model.forward_mut(input);
            total += loss.value(&output, target);
            model.backward_mut(&loss.gradient(&output, target));
        }
        optimizer.step(model, batch.len());
    }
    total / T::from_usize(data.len())
}

/// Mittlerer Verlust über alle Beispiele, ohne zu lernen.
///
/// # Panics
/// Wenn `data` leer ist.
pub fn mean_loss<M, L, T, I, Out>(model: &M, loss: &L, data: &[(I, Out)]) -> T
where
    T: Scalar,
    Out: Values<T>,
    M: ForwardPath<ForwardInput = I, ForwardOutput = Out>,
    L: Loss,
{
    assert!(!data.is_empty(), "data darf nicht leer sein");

    let total = data.iter().fold(T::ZERO, |total, (input, target)| {
        total + loss.value(&model.forward(input), target)
    });
    total / T::from_usize(data.len())
}
