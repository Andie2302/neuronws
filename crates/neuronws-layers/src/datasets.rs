//! Kleine Datensätze für Beispiele, Tests und eigene Experimente.
//!
//! Alle sind deterministisch (sie hängen nur vom übergebenen [`Rng`] ab) und kommen ohne `alloc`
//! aus: Die Größe steht als Zahl im Typ (`blobs::<120>`), das Ergebnis ist ein Array. So teilen
//! sich Beispiele und Tests dieselben Daten, statt sie je einmal zu kopieren.

use crate::random::Rng;
use crate::values::Image;

/// Die Wahrheitstabellen der Logikgatter als Lernbeispiele `(Eingabe, Ziel)`, so wie das
/// Perceptron sie lernt: zwei Eingänge, ein Ausgang als einzelne Zahl.
pub mod logic {
    /// Vier Beispiele mit zwei Eingängen und einem Ziel als einzelne Zahl.
    pub type Table = [([f32; 2], f32); 4];

    pub const AND: Table = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 0.0),
        ([1.0, 0.0], 0.0),
        ([1.0, 1.0], 1.0),
    ];

    pub const OR: Table = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 1.0),
    ];

    /// Das Gegenteil von UND: Nur bei zwei Einsen kommt 0 heraus. Mit `OR` zusammen ergibt es XOR.
    pub const NAND: Table = [
        ([0.0, 0.0], 1.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    /// Nicht linear trennbar: Kein einzelnes Neuron kann XOR lernen.
    pub const XOR: Table = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    /// Dieselbe Tabelle mit dem Ziel als Vektor aus einer Zahl, wie `Trainer` es braucht.
    pub const fn as_vectors(table: &Table) -> [([f32; 2], [f32; 1]); 4] {
        let mut result = [([0.0; 2], [0.0; 1]); 4];
        let mut i = 0;
        while i < 4 {
            result[i] = (table[i].0, [table[i].1]);
            i += 1;
        }
        result
    }
}

/// Die Mittelpunkte der drei Klassen von [`blobs`].
pub const BLOB_CENTERS: [[f32; 2]; 3] = [[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]];

/// Ein Beispiel von [`blobs`]: ein Punkt in der Ebene und die Klasse als One-Hot-Vektor.
pub type BlobSample = ([f32; 2], [f32; 3]);

/// `N` Punkte in drei Wolken um [`BLOB_CENTERS`], je `N / 3` pro Klasse, gestreut um höchstens
/// ±1,5 (jede Wolke füllt ein Quadrat der Seitenlänge 3). Die Reihenfolge ist gemischt, das Ziel
/// ist der One-Hot-Vektor der Klasse.
///
/// Die Klassen lassen sich trennen: Die Quadrate von Klasse 2 und den beiden anderen berühren sich
/// nur bei y = 0,5, zwischen Klasse 0 und 1 bleibt eine Lücke von 1. Sie liegen aber stellenweise
/// dicht beieinander, ein Netz mit wenigen Lernpunkten macht an den Grenzen deshalb noch einzelne
/// Fehler.
///
/// ```
/// use neuronws_layers::datasets::blobs;
/// use neuronws_layers::random::Rng;
///
/// let data = blobs::<120>(&mut Rng::new(42));
/// assert_eq!(data.len(), 120);
/// ```
///
/// # Panics
/// Wenn `N` nicht durch 3 teilbar ist.
pub fn blobs<const N: usize>(rng: &mut Rng) -> [BlobSample; N] {
    assert!(N.is_multiple_of(3), "N muss durch 3 teilbar sein");
    let per_class = N / 3;
    let mut data: [BlobSample; N] = core::array::from_fn(|i| {
        let class = i / per_class;
        let [cx, cy] = BLOB_CENTERS[class];
        let point = [cx + rng.uniform(-1.5, 1.5), cy + rng.uniform(-1.5, 1.5)];
        let mut target = [0.0; 3];
        target[class] = 1.0;
        (point, target)
    });
    rng.shuffle(&mut data);
    data
}

/// Die Klassen von [`lines`].
pub const LINE_CLASSES: [&str; 3] = ["senkrecht", "waagrecht", "diagonal"];

/// Ein 6×6-Bild mit einem Kanal, wie es [`lines`] liefert.
pub type LineImage = Image<1, 6, 6>;

/// Ein Beispiel von [`lines`]: das Bild und die Klasse als One-Hot-Vektor.
pub type LineSample = (LineImage, [f32; 3]);

/// Ein Bild der Klasse `class` (0 senkrecht, 1 waagrecht, 2 diagonal) mit der Linie an zufälliger
/// Stelle und Rauschen von ±0,2 auf jedem Pixel.
///
/// # Panics
/// Wenn `class` nicht kleiner als 3 ist.
pub fn line_sample(rng: &mut Rng, class: usize) -> LineSample {
    assert!(class < 3, "es gibt nur 3 Klassen");
    let mut image = [[[0.0_f32; 6]; 6]];
    let at = rng.next_below(6);
    for i in 0..6 {
        match class {
            0 => image[0][i][at] = 1.0,
            1 => image[0][at][i] = 1.0,
            _ if at.is_multiple_of(2) => image[0][i][i] = 1.0,
            _ => image[0][i][5 - i] = 1.0,
        }
    }
    for value in image[0].iter_mut().flatten() {
        *value += rng.uniform(-0.2, 0.2);
    }
    let mut target = [0.0; 3];
    target[class] = 1.0;
    (image, target)
}

/// `N` Bilder reihum aus den drei Klassen von [`LINE_CLASSES`]: senkrechte, waagrechte und
/// diagonale Linie in verrauschten 6×6-Bildern, gemischt. Weil die Linie an zufälliger Stelle
/// liegt, muss ein Netz sie überall erkennen. Genau die Aufgabe, bei der Faltung hilft.
///
/// ```
/// use neuronws_layers::datasets::lines;
/// use neuronws_layers::random::Rng;
///
/// let data = lines::<60>(&mut Rng::new(7));
/// assert_eq!(data.len(), 60);
/// ```
pub fn lines<const N: usize>(rng: &mut Rng) -> [LineSample; N] {
    let mut data: [LineSample; N] = core::array::from_fn(|i| line_sample(rng, i % 3));
    rng.shuffle(&mut data);
    data
}
