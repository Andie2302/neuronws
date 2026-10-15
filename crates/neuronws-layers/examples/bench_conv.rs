//! Wie schnell ist die Faltung? Misst Vorwärts- und Rückwärtsrechnung einiger typischer Größen.
//!
//! Die Faltung ist mit einfachen Schleifen geschrieben, ohne "im2col" und ohne Matrix-Tricks. Für
//! kleine Bilder ist das schnell genug, für größere lohnt es sich, nachzumessen. Das Programm
//! gibt für jede Größe die Zeit pro Aufruf und die Rechenleistung in Millionen
//! Multiplikation-Additionen pro Sekunde (MMAC/s) aus. Beim Rückwärtsrechnen sind es etwa doppelt
//! so viele Rechenschritte wie vorwärts (Gradient der Kerne und Gradient der Eingabe), das ist in
//! der Spalte schon eingerechnet.
//!
//! Nur im Release-Build aussagekräftig:
//!
//! ```text
//! cargo run -p neuronws-layers --features libm --release --example bench_conv
//! ```
//!
//! # Was die Messung gebracht hat
//!
//! Die erste Fassung rechnete für jede Ausgabestelle alle Kernpunkte durch und prüfte dabei bei
//! jedem Schritt, ob der Punkt im Bild oder im Rand liegt. Die jetzige Fassung dreht das um: Die
//! Kernpunkte laufen außen, und für jedes Gewicht wird ein ganzer **Streifen** der Ausgabezeile in
//! einem Zug aktualisiert. Der Rand wird einmal pro Streifen ausgerechnet statt bei jedem Schritt,
//! und die innere Schleife läuft über zusammenhängenden Speicher, den der Compiler in
//! Vektorbefehle übersetzt. Rückwärts gilt dasselbe für breite Zeilen. Bei schmalen Zeilen
//! (unter 16 Ausgabespalten) ist die einfache Schleife schneller, weil der Aufwand pro Streifen
//! sich dort nicht lohnt.
//!
//! Gemessen auf dem Rechner, auf dem das geschrieben wurde (x86-64, Release). Absolute Zahlen
//! hängen vom Rechner ab, die Verhältnisse nicht:
//!
//! ```text
//!                                         vorwärts                 rückwärts
//! Faltung                            vorher   nachher       vorher   nachher
//! 1→8 Kanäle, 28×28, 5×5, valid       42 µs    12 µs         37 µs    38 µs
//! 8→16 Kanäle, 12×12, 5×5, valid     105 µs    14 µs         59 µs    54 µs
//! 3→16 Kanäle, 32×32, 3×3, same      258 µs    50 µs        401 µs   195 µs
//! 16→32 Kanäle, 16×16, 3×3, same     626 µs   138 µs        872 µs   640 µs
//! 16→32 Kanäle, 16×16, Schritt 2     156 µs   138 µs        129 µs   126 µs
//! ```
//!
//! Vorwärts ist die Faltung bei Schrittweite 1 drei- bis siebenmal so schnell, rückwärts bis zu
//! doppelt so schnell. Bei Schrittweite 2 liegen die Werte nicht nebeneinander im Speicher, dort
//! bleibt alles beim Alten. Jede Änderung wurde gegen die Gradienten-Prüfung und den unabhängigen
//! Vergleich in `tests/conv_geometry.rs` abgesichert: Beim Umbau hat genau diese Prüfung einen
//! Fehler in der Restbehandlung der Teilsummen gefunden.
//!
//! Was weiterhin fehlt: "im2col" (die Bildausschnitte als Matrix anordnen und alles als eine große
//! Matrixmultiplikation rechnen) und kachelweises Rechnen für den Cache. Beides lohnt sich erst bei
//! deutlich größeren Bildern und Kanalzahlen, als eine Bibliothek ohne Heap sinnvoll auf den Stack
//! bekommt.

use std::hint::black_box;
use std::time::{Duration, Instant};

use neuronws_layers::block::Operation;
use neuronws_layers::conv::{Conv2d, Kernels};
use neuronws_layers::values::{self, Image};

/// Führt `f` mindestens 400 ms lang aus und liefert die mittlere Zeit pro Aufruf.
fn time(mut f: impl FnMut()) -> Duration {
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    let mut runs = 0_u32;
    while start.elapsed() < Duration::from_millis(400) {
        f();
        runs += 1;
    }
    start.elapsed() / runs
}

/// Misst eine Faltung und gibt eine Zeile aus.
fn bench<
    const C_IN: usize,
    const C_OUT: usize,
    const H: usize,
    const W: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
    const OH: usize,
    const OW: usize,
>(
    name: &str,
) {
    let op = Conv2d::<C_IN, C_OUT, H, W, KH, KW, STRIDE, PAD, 1, OH, OW>::new();
    let kernels: Kernels<C_OUT, C_IN, KH, KW> =
        values::from_fn(|i| ((i * 7 % 13) as f32 - 6.0) * 0.05);
    let params = (kernels, [0.1_f32; C_OUT]);
    let input: Image<C_IN, H, W> = values::from_fn(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1);
    let grad_output: Image<C_OUT, OH, OW> = values::from_fn(|i| ((i * 3 % 7) as f32 - 3.0) * 0.1);
    let output = op.forward(&params, &input);

    let forward = time(|| {
        black_box(op.forward(black_box(&params), black_box(&input)));
    });

    let mut grads: (Kernels<C_OUT, C_IN, KH, KW>, [f32; C_OUT]) =
        ([[[[0.0; KW]; KH]; C_IN]; C_OUT], [0.0; C_OUT]);
    let backward = time(|| {
        black_box(op.backward(
            black_box(&params),
            &mut grads,
            black_box(&input),
            black_box(&output),
            black_box(&grad_output),
        ));
    });

    let macs = (C_OUT * OH * OW * C_IN * KH * KW) as f64;
    println!(
        "{name:36} {:>10.1?} {:>8.0}   {:>10.1?} {:>8.0}",
        forward,
        macs / forward.as_secs_f64() / 1e6,
        backward,
        2.0 * macs / backward.as_secs_f64() / 1e6,
    );
}

fn main() {
    println!(
        "{:36} {:>10} {:>8}   {:>10} {:>8}",
        "Faltung", "vorwärts", "MMAC/s", "rückwärts", "MMAC/s"
    );
    bench::<1, 8, 28, 28, 5, 5, 1, 0, 24, 24>("1→8 Kanäle, 28×28, 5×5, valid");
    bench::<8, 16, 12, 12, 5, 5, 1, 0, 8, 8>("8→16 Kanäle, 12×12, 5×5, valid");
    bench::<3, 16, 32, 32, 3, 3, 1, 1, 32, 32>("3→16 Kanäle, 32×32, 3×3, same");
    bench::<16, 32, 16, 16, 3, 3, 1, 1, 16, 16>("16→32 Kanäle, 16×16, 3×3, same");
    bench::<16, 32, 16, 16, 3, 3, 2, 1, 8, 8>("16→32 Kanäle, 16×16, 3×3, Schritt 2");
}
