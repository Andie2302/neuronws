//! Wie weit zurück kann sich ein Netz erinnern? RNN, GRU und LSTM im Vergleich.
//!
//! Start (im Release-Build rund 20 Sekunden, im Debug-Build ein Vielfaches davon):
//!
//! ```text
//! cargo run -p neuronws-layers --features libm --release --example sequence_memory
//! ```
//!
//! # Die Aufgabe
//!
//! Eine Folge aus lauter −1 und 1 läuft ein, Wert für Wert. Am Ende soll das Netz sagen, welcher
//! Wert **am Anfang** stand. Alles dazwischen ist Rauschen, das es ignorieren muss. Ohne Gedächtnis
//! geht das nicht: Wer nur den letzten Wert sieht, rät und liegt in der Hälfte der Fälle richtig.
//! Je länger die Folge, desto weiter muss sich das Netz zurückerinnern.
//!
//! # Warum das schwer ist
//!
//! Beim Lernen läuft der Fehler rückwärts durch die Zeit (BPTT): Der Gradient für den ersten
//! Schritt geht durch jeden späteren Schritt, und in jedem wird er mit den Gewichten der
//! Rückkopplung multipliziert. Ist dieser Faktor kleiner als 1, schrumpft der Gradient mit jedem
//! Schritt (bei 0,9 sind nach 30 Schritten nur noch 4 % übrig), ist er größer, wächst er. Der
//! verschwindende Gradient macht es dem einfachen RNN schwer, lange Abhängigkeiten zu lernen.
//! GRU und LSTM haben dagegen Tore, die entscheiden, wie viel vom alten Zustand bleibt. Der alte
//! Zustand läuft dann **ohne Multiplikation mit Gewichten** weiter, nur mit dem Wert des Tors
//! (zwischen 0 und 1). Steht das Tor weit offen (nahe 1), verschwindet der Gradient auf diesem
//! Weg kaum, bei halb offenem Tor (0,5) verliert er in jedem Schritt die Hälfte. Wie weit die Tore
//! am Anfang offen stehen, zeigt die zweite Tabelle.
//!
//! # Was gemessen wird
//!
//! **Tabelle 1:** Für jede Länge der Folge und jedes Netz der Anteil der neuen Folgen, bei denen
//! das Vorzeichen der Ausgabe stimmt, nach 150 Runden mit AdamW. Mittel aus 4 Zufallsstarts, die
//! Starts geben allen Netzen dieselben Daten. Die erste Spalte ist ein Netz, das nur den letzten
//! Wert sieht. Das RNN gibt es zweimal, mit orthogonalen und mit Xavier-Startwerten (siehe
//! `Orthogonal` in `neuronws_layers::init`).
//!
//! **Tabelle 2:** Bei Länge 32 der Bias des Update-Tors der GRU und des Vergiss-Tors des LSTM, der
//! am Anfang des Lernens gesetzt wird. Er bestimmt, wie viel vom alten Zustand ein Schritt
//! durchlässt.
//!
//! # Was die Messung zeigt
//!
//! ```text
//!               ohne Gedächtnis   RNN orthogonal   RNN Xavier     GRU      LSTM
//! Länge  4          49.9 %          100.0 %         100.0 %    100.0 %   100.0 %
//! Länge  8          49.9 %          100.0 %         100.0 %    100.0 %   100.0 %
//! Länge 16          48.1 %           99.8 %          88.2 %     96.9 %    99.7 %
//! Länge 32          49.4 %           73.8 %          86.2 %     54.3 %    71.4 %
//!
//! Länge 32, Bias des Tors am Anfang:   0        1        2        3
//!   GRU (Update-Tor)                 54.3 %   63.2 %   99.9 %   99.8 %
//!   LSTM (Vergiss-Tor)               47.5 %   71.4 %   95.3 %   88.3 %
//! ```
//!
//! 1. **Ohne Gedächtnis bleibt es beim Raten** (rund 50 %), egal wie lang die Folge ist. Mit
//!    Gedächtnis ist die Aufgabe bis 8 Schritte für jedes Netz gelöst.
//! 2. **Die Startwerte des einfachen RNN entscheiden.** Bei 16 Schritten schafft es mit
//!    orthogonalen Startwerten 99,8 %, mit Xavier nur 88,2 %. Eine orthogonale Matrix lässt die
//!    Länge eines Vektors unverändert, die Rückkopplung lässt den Gradienten also nicht schon
//!    durch ihre Gewichte schrumpfen oder wachsen (das Tanh dämpft ihn weiterhin etwas). Das ist
//!    der Grund für die Empfehlung in `recurrent`.
//! 3. **Bei 32 Schritten wird es bei allen unzuverlässig.** Ein einzelner Start gelingt dort
//!    meist ganz (fast 100 %) oder gar nicht (um 50 %). Ein Mittel wie 73,8 % heißt: Zwei von
//!    vier Starts sind gelungen. Die Unterschiede zwischen den Spalten (73,8 gegen 86,2 %) sind
//!    bei 4 Starts nicht zu deuten, sie sind einzelne Starts mehr oder weniger.
//! 4. **Tore helfen nur, wenn sie am Anfang offen stehen.** Mit Bias 0 lässt das Update-Tor der
//!    GRU in jedem Schritt die Hälfte des alten Zustands durch (σ(0) = 0,5). Vom ersten Wert
//!    bleibt nach 31 Schritten 0,5³¹ ≈ 5·10⁻¹⁰ übrig, weniger als die Rechengenauigkeit von
//!    `f32`, und der Gradient, der bis zum ersten Schritt zurückfließt, ist ebenso winzig. Mit
//!    Bias 2 (σ(2) = 0,88) bleiben 0,02, mit Bias 3 (σ(3) = 0,95) etwa 0,2. Die GRU lernt dann in
//!    allen vier Starts (99,9 und 99,8 %). Das LSTM startet schon mit Bias 1 (σ(1) = 0,73, nach
//!    31 Schritten 6·10⁻⁵). Das passt dazu, dass es hier besser abschneidet als die GRU mit
//!    Bias 0 (71,4 gegen 54,3 %), mehr als ein Hinweis ist es bei 4 Starts aber nicht. Mit Bias 2
//!    kommt das LSTM auf 95,3 %. Ob der Rückgang bei Bias 3 (88,3 %) mehr als Zufall ist, sagen
//!    4 Starts nicht.
//!
//! Was man mitnehmen kann: Dass GRU und LSTM "Tore haben, die das Gedächtnis schützen", stimmt
//! erst, wenn die Tore am Anfang auch offen sind. Mit der Standardeinstellung allein gewinnen sie
//! hier nicht gegen ein gut gestartetes RNN. Die Zahlen gelten für das Budget von 150 Runden: Wer
//! in `learn` 400 statt 150 einsetzt, sieht, dass auch die schlecht gestarteten Netze dann
//! aufholen (bei der Erstellung dieses Textes: GRU mit Bias 0 von 54 auf 86 %).

use neuronws_layers::activation::Tanh;
use neuronws_layers::adam::AdamW;
use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::init::{Orthogonal, Xavier};
use neuronws_layers::loss::MeanSquaredError;
use neuronws_layers::random::Rng;
use neuronws_layers::recurrent::{Gru, Lstm, Rnn};
use neuronws_layers::sequence::LastStepLayer;
use neuronws_layers::trainer::{Learner, Trainer};
use neuronws_layers::traits::{ForwardPath, Trainable};

/// Größe des Zustands (und damit des Gedächtnisses) der Zellen.
const HIDDEN: usize = 8;
/// Wie viele Zufallsstarts in jede Zahl der Tabellen eingehen.
const STARTS: u64 = 4;

type Sequence<const STEPS: usize> = [[f32; 1]; STEPS];
type Sample<const STEPS: usize> = (Sequence<STEPS>, [f32; 1]);

/// Eine Folge aus lauter −1 und 1. Das Ziel ist der erste Wert.
fn make_sample<const STEPS: usize>(rng: &mut Rng) -> Sample<STEPS> {
    let mut sequence = [[0.0; 1]; STEPS];
    for step in &mut sequence {
        step[0] = if rng.next_f32() < 0.5 { -1.0 } else { 1.0 };
    }
    (sequence, [sequence[0][0]])
}

/// Anteil der Folgen, bei denen die Ausgabe das richtige Vorzeichen hat.
fn sign_accuracy<const STEPS: usize>(
    net: &impl ForwardPath<ForwardInput = Sequence<STEPS>, ForwardOutput = [f32; 1]>,
    data: &[Sample<STEPS>],
) -> f32 {
    let right = data
        .iter()
        .filter(|(x, target)| net.forward(x)[0] * target[0] > 0.0)
        .count();
    right as f32 / data.len() as f32
}

/// Lernt 150 Runden und liefert die Trefferquote auf neuen Folgen.
fn learn<const STEPS: usize, M>(mut net: M, seed: u64) -> f32
where
    M: Learner<Sequence<STEPS>, [f32; 1]>
        + ForwardPath<ForwardInput = Sequence<STEPS>, ForwardOutput = [f32; 1]>
        + Trainable<Scalar = f32>,
{
    let mut rng = Rng::new(seed);
    let train: Vec<Sample<STEPS>> = (0..96).map(|_| make_sample(&mut rng)).collect();
    let test: Vec<Sample<STEPS>> = (0..300).map(|_| make_sample(&mut rng)).collect();
    let mut trainer = Trainer::new(MeanSquaredError, AdamW::for_model(&net, 0.02, 0.0), 16);
    for _ in 0..150 {
        let _loss: f32 = trainer.epoch(&mut net, &train);
    }
    sign_accuracy(&net, &test)
}

/// Mittlere Trefferquote in Prozent über alle Zufallsstarts. `build` baut das Netz, die Daten
/// kommen aus einem eigenen Zufallsstrom und sind deshalb für jedes Netz dieselben.
fn average<const STEPS: usize, M>(build: impl Fn(&mut Rng) -> M) -> f32
where
    M: Learner<Sequence<STEPS>, [f32; 1]>
        + ForwardPath<ForwardInput = Sequence<STEPS>, ForwardOutput = [f32; 1]>
        + Trainable<Scalar = f32>,
{
    let sum: f32 = (1..=STARTS)
        .map(|seed| learn::<STEPS, M>(build(&mut Rng::new(seed)), 1000 + seed))
        .sum();
    100.0 * sum / STARTS as f32
}

/// Eine Zeile der ersten Tabelle: fünf Netze für Folgen der Länge `STEPS`.
fn row<const STEPS: usize>() {
    // Nur der letzte Wert zählt: ein Netz ohne Gedächtnis.
    let blind = average::<STEPS, _>(|rng| {
        LastStepLayer::<STEPS, 1>::new().then(DenseLayer::<1, 1>::xavier(rng))
    });
    let rnn_orthogonal = average::<STEPS, _>(|rng| {
        Block::initialized(Rnn::<STEPS, 1, HIDDEN, _>::new(Tanh), &Orthogonal(1.0), rng)
            .then(LastStepLayer::<STEPS, HIDDEN>::new())
            .then(DenseLayer::<HIDDEN, 1>::xavier(rng))
    });
    let rnn_xavier = average::<STEPS, _>(|rng| {
        Block::initialized(Rnn::<STEPS, 1, HIDDEN, _>::new(Tanh), &Xavier, rng)
            .then(LastStepLayer::<STEPS, HIDDEN>::new())
            .then(DenseLayer::<HIDDEN, 1>::xavier(rng))
    });
    let gru = average::<STEPS, _>(|rng| {
        Block::initialized(Gru::<STEPS, 1, HIDDEN>::new(), &Xavier, rng)
            .then(LastStepLayer::<STEPS, HIDDEN>::new())
            .then(DenseLayer::<HIDDEN, 1>::xavier(rng))
    });
    let lstm = average::<STEPS, _>(|rng| {
        Block::initialized(Lstm::<STEPS, 1, HIDDEN>::new(), &Xavier, rng)
            .then(LastStepLayer::<STEPS, HIDDEN>::new())
            .then(DenseLayer::<HIDDEN, 1>::xavier(rng))
    });
    print!("  {STEPS:6} |");
    for accuracy in [blind, rnn_orthogonal, rnn_xavier, gru, lstm] {
        print!(" {:>15} |", format!("{accuracy:.1} %"));
    }
    println!();
}

/// Eine Zeile der zweiten Tabelle: GRU und LSTM für Folgen der Länge `STEPS`, deren Tor am Anfang
/// mit dem Bias `bias` startet.
fn gate_row<const STEPS: usize>(bias: f32) {
    let gru = average::<STEPS, _>(|rng| {
        let mut gru = Block::initialized(Gru::<STEPS, 1, HIDDEN>::new(), &Xavier, rng);
        // Die Parameter der GRU sind `((Rücksetz-Tor, Update-Tor), Kandidat)`, ein Tor ist
        // `((Wx, Wh), Bias)`. Der Bias des Update-Tors steht also bei `.0.1.1`.
        let mut parameters = gru.parameters();
        parameters.0.1.1 = [bias; HIDDEN];
        gru.set_parameters(&parameters);
        gru.then(LastStepLayer::<STEPS, HIDDEN>::new())
            .then(DenseLayer::<HIDDEN, 1>::xavier(rng))
    });
    let lstm = average::<STEPS, _>(|rng| {
        let mut lstm = Block::initialized(Lstm::<STEPS, 1, HIDDEN>::new(), &Xavier, rng);
        // Die Parameter des LSTM sind `((Eingabe-Tor, Vergiss-Tor), (Kandidat, Ausgabe-Tor))`. Der
        // Bias des Vergiss-Tors steht bei `.0.1.1`.
        let mut parameters = lstm.parameters();
        parameters.0.1.1 = [bias; HIDDEN];
        lstm.set_parameters(&parameters);
        lstm.then(LastStepLayer::<STEPS, HIDDEN>::new())
            .then(DenseLayer::<HIDDEN, 1>::xavier(rng))
    });
    // Die Standardwerte der Bibliothek: Update-Tor 0, Vergiss-Tor 1.
    let gru_mark = if bias == 0.0 { '*' } else { ' ' };
    let lstm_mark = if bias == 1.0 { '*' } else { ' ' };
    let gru = format!("{gru:.1} %{gru_mark}");
    let lstm = format!("{lstm:.1} %{lstm_mark}");
    println!("  {bias:4} | {gru:>14} | {lstm:>14} |");
}

fn main() {
    println!(
        "1. Richtig erinnerter erster Wert (Mittel aus {STARTS} Starts, Zustand aus {HIDDEN} Zahlen)"
    );
    print!("  Länge |");
    for name in [
        "ohne Gedächtnis",
        "RNN orthogonal",
        "RNN Xavier",
        "GRU",
        "LSTM",
    ] {
        print!(" {name:^15} |");
    }
    println!();
    row::<4>();
    row::<8>();
    row::<16>();
    row::<32>();

    println!(
        "\n2. Länge 32: Wie weit stehen die Tore am Anfang offen? (* = Standard der Bibliothek)"
    );
    println!(
        "  Bias | {:^14} | {:^14} |",
        "GRU (Update)", "LSTM (Vergiss)"
    );
    for bias in [0.0, 1.0, 2.0, 3.0] {
        gate_row::<32>(bias);
    }
}
