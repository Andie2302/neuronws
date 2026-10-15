//! Welcher Optimierer lernt am besten? SGD, Momentum, Nesterov, RMSprop, AdaGrad, Adam und AdamW
//! im Vergleich, auf drei Aufgaben und mit je acht Lernraten.
//!
//! Start (im Release-Build dauert es eine Sekunde, im Debug-Build eine halbe Minute):
//!
//! ```text
//! cargo run -p neuronws-layers --features libm --release --example optimizer_comparison
//! ```
//!
//! # Wie fair verglichen wird
//!
//! Ein Vergleich von Optimierern ist nur so gut wie seine Lernraten. Mit *einer* Lernrate für alle
//! gewinnt zufällig, wer gerade dazu passt. Deshalb bekommt hier jeder Optimierer dieselben acht
//! Lernraten von 0,001 bis 3, und die Tabelle zeigt für jede Aufgabe den Verlust bei **jeder**
//! davon. Der Stern markiert die beste Lernrate einer Zeile. So sieht man zweierlei: wie gut ein
//! Optimierer *bestenfalls* ist (der Stern), und wie *wählerisch* er bei der Lernrate ist (wie breit
//! der Bereich um den Stern ist, in dem er noch brauchbar lernt).
//!
//! - **Gleiche Bedingungen:** Jede Zelle ist der Mittelwert über 8 Zufallsstarts. Der Start `s` gibt
//!   allen Optimierern dieselben Daten und dieselben Anfangsgewichte, nur die Lernregel
//!   unterscheidet sich. Die Zahlen sind deshalb bei jedem Aufruf dieselben.
//! - **Festes Budget:** Gemessen wird der Verlust auf frischen Prüfdaten nach einer festen Zahl von
//!   Runden. Wer schneller lernt, hat dann den kleineren Verlust.
//! - `div.` heißt: Das Lernen ist bei mindestens einem Start explodiert (Verlust unendlich, `NaN`
//!   oder über 100).
//!
//! Die übrigen Einstellungen sind die üblichen Standardwerte (Impuls 0,9; RMSprop-Zerfall 0,9;
//! Adam mit β₁ = 0,9 und β₂ = 0,999; AdamW mit Gewichtszerfall 0,01) und werden nicht pro Aufgabe
//! nachgestellt.
//!
//! # Wie der Code gebaut ist
//!
//! Jeder Optimierer ist ein eigener Typ, also lassen sich die sieben nicht in eine Liste stecken.
//! Statt den Aufruf siebenmal abzuschreiben, bekommt jede Sorte ein kleines Trait `Family` ("wie
//! baue ich dich für dieses Netz und diese Lernrate?") und jede Aufgabe ein Trait `Task`
//! ("trainiere ein Netz mit einer Sorte und liefere den Verlust"). Die Funktion `row` ist generisch
//! über beides, der Compiler erzeugt daraus die 21 Kombinationen.
//!
//! # Was die Messung zeigt
//!
//! Der beste Verlust jeder Zeile, mit der Lernrate in Klammern (die Tabellen beim Start zeigen alle
//! Lernraten):
//!
//! ```text
//!           XOR                 Punktwolken         Sinuskurve
//! SGD       0.0026  (Rate 3)    0.0283  (Rate 1)    0.0119  (Rate 0.03)
//! Momentum  0.0000  (Rate 3)    0.0287  (Rate 0.1)  0.0077  (Rate 0.01)
//! Nesterov  0.0005  (Rate 1)    0.0282  (Rate 0.1)  0.0070  (Rate 0.03)
//! RMSprop   0.0001  (Rate 0.03) 0.0285  (Rate 0.01) 0.0107  (Rate 0.003)
//! AdaGrad   0.0276  (Rate 0.3)  0.0249  (Rate 0.3)  0.0097  (Rate 0.3)
//! Adam      0.0010  (Rate 0.1)  0.0260  (Rate 0.03) 0.0048  (Rate 0.01)
//! AdamW     0.0018  (Rate 0.3)  0.0240  (Rate 0.1)  0.0056  (Rate 0.01)
//! ```
//!
//! 1. **Mit der richtigen Lernrate liegen die Optimierer dichter beieinander, als ihr Ruf vermuten
//!    lässt.** Auf den Punktwolken landen alle sieben zwischen 0,024 und 0,029. Auf der Sinuskurve
//!    liegt der beste (Adam, 0,0048) um den Faktor 2,5 vor SGD (0,0119). Bei XOR lösen alle sieben
//!    die Aufgabe (Verlust unter 0,03), die Unterschiede liegen dort nur in der Genauigkeit.
//! 2. **Die beste Lernrate unterscheidet sich um Größenordnungen.** SGD wollte bei XOR und
//!    Punktwolken 1 bis 3, Adam 0,03 bis 0,1. Dieselbe Rate 0,1 bringt XOR mit Adam auf 0,0010 und
//!    mit SGD nur auf 0,45, also kaum gelernt. Wer Optimierer mit einer gemeinsamen Rate vergleicht,
//!    vergleicht vor allem die Raten. Die oft genannte Standardrate 0,001 für Adam ist bei diesen
//!    kleinen Netzen und wenigen Schritten zu klein (erste Spalte); 0,01 bis 0,1 passte besser.
//! 3. **Das brauchbare Fenster ist bei allen etwa gleich breit, es liegt nur woanders.** Zählt man
//!    die Lernraten, bei denen ein Optimierer höchstens dreimal so schlecht ist wie bei seiner
//!    besten, sind es auf den Punktwolken und der Sinuskurve bei jedem drei bis fünf von acht, also
//!    mehr als eine Größenordnung. Adam hat hier kein breiteres Fenster als SGD, es liegt nur
//!    tiefer (Punktwolken: 0,01 bis 0,3 gegen 0,1 bis 3 bei SGD).
//! 4. **Zu große Raten scheitern verschieden.** SGD, Momentum und Nesterov explodieren auf der
//!    Sinuskurve ab 0,3 (`div.`): Ein zu großer Schritt macht den Gradienten größer, der macht den
//!    nächsten Schritt größer, und so weiter. RMSprop, Adam und AdamW teilen jeden Schritt durch
//!    die typische Größe des Gradienten, ein einzelner Schritt bleibt daher in der Größenordnung
//!    der Lernrate, egal wie groß der Gradient wird. Bei ihnen wird der Verlust zuerst nur
//!    schlechter (Adam bei 0,3: 0,030, bei 1: 0,79), RMSprop explodiert erst bei 3. In diesen
//!    Messungen liegt der Vorteil der adaptiven Verfahren eher in dieser Gutmütigkeit nach oben als
//!    in der Bestleistung.
//! 5. **Impuls bringt viel, Nesterov kaum etwas zusätzlich.** Bei Rate 0,1 erreicht Momentum auf XOR
//!    0,0097, SGD nur 0,45 (der Impuls 0,9 macht die Schritte bis zu zehnmal so groß). Auf der
//!    Sinuskurve ist Momentum mit 0,0077 besser als SGD mit 0,0119. Nesterov liegt nur wenige
//!    Prozent neben Momentum (Punktwolken: 0,0282 gegen 0,0287, Sinuskurve: 0,0070 gegen 0,0077),
//!    auf so kleinen Aufgaben ist das kein Grund zu wechseln.
//! 6. **AdaGrad ist in keiner Zelle explodiert, braucht aber große Raten.** Es summiert alle
//!    bisherigen Quadrate der Gradienten auf, seine Schritte werden also nur kleiner. Rate 0,3 war
//!    auf allen drei Aufgaben die beste. Bei kleinen Raten lernt es auf den Punktwolken und der
//!    Sinuskurve am langsamsten (Sinuskurve bei 0,001: 0,35, SGD: 0,096).
//! 7. **Speicher:** SGD braucht nichts außer der Lernrate, Momentum, Nesterov, RMSprop und AdaGrad
//!    je eine Zahl pro Parameter, Adam und AdamW zwei. Bei einem Netz mit 10 000 Parametern sind
//!    das 40 kB für die Parameter und weitere 80 kB für Adam. Auf einem Mikrocontroller mit wenig
//!    RAM entscheidet das, dort sind SGD oder Momentum der vernünftige Kompromiss.
//!
//! Was die Messung **nicht** sagt: Die Aufgaben sind klein, die Rundenzahl ist fest und es gibt
//! keinen Lernraten-Plan. Bei großen Netzen, langem Training oder mit einem Plan können die
//! Ergebnisse anders aussehen. Kleine Unterschiede zwischen Nachbarzellen sind Rauschen, gelten
//! sollten nur die großen Muster. Steht der Stern in der letzten Spalte (SGD und Momentum auf XOR),
//! liegt die beste Rate vermutlich noch dahinter.

use std::mem::size_of_val;

use neuronws_layers::activation::{Sigmoid, Tanh};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::adam::{Adam, AdamW};
use neuronws_layers::adaptive::{AdaGrad, RmsProp};
use neuronws_layers::chain::LayerExt;
use neuronws_layers::datasets::{blobs, logic};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::{BinaryCrossEntropy, CategoricalCrossEntropy, MeanSquaredError};
use neuronws_layers::optimizer::{Momentum, Optimizer, Sgd};
use neuronws_layers::random::Rng;
use neuronws_layers::softmax_layer::SoftmaxLayer;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::Trainable;

/// Die Lernraten, mit denen jeder Optimierer antritt.
const RATES: [f32; 8] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0];

/// Wie viele Zufallsstarts in jede Zelle eingehen.
const SEEDS: u64 = 8;

/// Eine Sorte von Optimierer, die sich für jedes Netz und jede Lernrate bauen lässt.
trait Family {
    const NAME: &'static str;

    /// Der Optimierer für `net`. Das `use<M>` sagt, dass das Ergebnis nur vom Typ `M` abhängt und
    /// nicht den geliehenen `net` festhält (seit Edition 2024 würde `impl Trait` sonst die Leihe
    /// mitnehmen, und das Netz wäre während des Lernens gesperrt).
    fn build<M: Trainable<Scalar = f32>>(net: &M, rate: f32) -> impl Optimizer<M> + use<M, Self>;
}

struct SgdKind;
impl Family for SgdKind {
    const NAME: &'static str = "SGD";
    fn build<M: Trainable<Scalar = f32>>(net: &M, rate: f32) -> impl Optimizer<M> + use<M> {
        Sgd::for_model(net, rate)
    }
}

struct MomentumKind;
impl Family for MomentumKind {
    const NAME: &'static str = "Momentum";
    fn build<M: Trainable<Scalar = f32>>(net: &M, rate: f32) -> impl Optimizer<M> + use<M> {
        Momentum::for_model(net, rate, 0.9)
    }
}

struct NesterovKind;
impl Family for NesterovKind {
    const NAME: &'static str = "Nesterov";
    fn build<M: Trainable<Scalar = f32>>(net: &M, rate: f32) -> impl Optimizer<M> + use<M> {
        Momentum::for_model(net, rate, 0.9).with_nesterov()
    }
}

struct RmsPropKind;
impl Family for RmsPropKind {
    const NAME: &'static str = "RMSprop";
    fn build<M: Trainable<Scalar = f32>>(net: &M, rate: f32) -> impl Optimizer<M> + use<M> {
        RmsProp::for_model(net, rate)
    }
}

struct AdaGradKind;
impl Family for AdaGradKind {
    const NAME: &'static str = "AdaGrad";
    fn build<M: Trainable<Scalar = f32>>(net: &M, rate: f32) -> impl Optimizer<M> + use<M> {
        AdaGrad::for_model(net, rate)
    }
}

struct AdamKind;
impl Family for AdamKind {
    const NAME: &'static str = "Adam";
    fn build<M: Trainable<Scalar = f32>>(net: &M, rate: f32) -> impl Optimizer<M> + use<M> {
        Adam::for_model(net, rate)
    }
}

struct AdamWKind;
impl Family for AdamWKind {
    const NAME: &'static str = "AdamW";
    fn build<M: Trainable<Scalar = f32>>(net: &M, rate: f32) -> impl Optimizer<M> + use<M> {
        AdamW::for_model(net, rate, 0.01)
    }
}

/// Eine Aufgabe: ein Netz mit Daten und Fehlerfunktion, bei dem nur der Optimierer offen bleibt.
trait Task {
    const NAME: &'static str;
    const DETAILS: &'static str;

    /// Trainiert ein frisches Netz (Daten und Startgewichte aus `seed`) mit der Sorte `F` und der
    /// Lernrate `rate`. Liefert den Verlust auf Daten, die das Netz nie gesehen hat.
    fn run<F: Family>(seed: u64, rate: f32) -> f32;
}

/// XOR: vier Beispiele, ein winziges Netz. Prüfdaten sind dieselben vier Punkte.
struct Xor;
impl Task for Xor {
    const NAME: &'static str = "XOR";
    const DETAILS: &'static str =
        "2→4 (Tanh)→1 (Sigmoid), Kreuzentropie, alle 4 Beispiele pro Schritt, 300 Runden";

    fn run<F: Family>(seed: u64, rate: f32) -> f32 {
        let mut rng = Rng::new(seed);
        let mut net = DenseLayer::<2, 4>::xavier(&mut rng)
            .then(ActivationLayer::<4, _>::new(Tanh))
            .then(DenseLayer::<4, 1>::xavier(&mut rng))
            .then(ActivationLayer::<1, _>::new(Sigmoid));
        let data = logic::as_vectors(&logic::XOR);
        let mut trainer = Trainer::new(BinaryCrossEntropy, F::build(&net, rate), 4);
        trainer.fit(&mut net, &data, 300);
        trainer.evaluate(&net, &data)
    }
}

/// Drei Punktwolken, die dicht beieinander liegen: Klassifikation mit Softmax.
struct Blobs;
impl Task for Blobs {
    const NAME: &'static str = "Drei Punktwolken";
    const DETAILS: &'static str = "2→8 (Tanh)→3 (Softmax), Kreuzentropie, 120 Lernpunkte, Mini-Batches zu 16, 40 Runden; \
         Verlust auf 300 neuen Punkten";

    fn run<F: Family>(seed: u64, rate: f32) -> f32 {
        let mut rng = Rng::new(seed);
        let train = blobs::<120>(&mut rng);
        let test = blobs::<300>(&mut rng);
        let mut net = DenseLayer::<2, 8>::xavier(&mut rng)
            .then(ActivationLayer::<8, _>::new(Tanh))
            .then(DenseLayer::<8, 3>::xavier(&mut rng))
            .then(SoftmaxLayer::<3>::new());
        let mut trainer = Trainer::new(CategoricalCrossEntropy, F::build(&net, rate), 16);
        trainer.fit(&mut net, &train, 40);
        trainer.evaluate(&net, &test)
    }
}

/// Eine Kurve nachzeichnen: Regression mit dem mittleren quadratischen Fehler.
struct Sine;
impl Task for Sine {
    const NAME: &'static str = "Sinuskurve";
    const DETAILS: &'static str = "1→12 (Tanh)→1, quadratischer Fehler, 64 verrauschte Punkte von sin(x) auf [−3, 3], \
         Mini-Batches zu 8, 150 Runden; Fehler auf 101 sauberen Punkten";

    fn run<F: Family>(seed: u64, rate: f32) -> f32 {
        let mut rng = Rng::new(seed);
        let train: [([f32; 1], [f32; 1]); 64] = std::array::from_fn(|_| {
            let x = rng.uniform(-3.0, 3.0);
            ([x], [x.sin() + rng.uniform(-0.1, 0.1)])
        });
        let test: [([f32; 1], [f32; 1]); 101] = std::array::from_fn(|i| {
            let x = -3.0 + 6.0 * i as f32 / 100.0;
            ([x], [x.sin()])
        });
        let mut net = DenseLayer::<1, 12>::xavier(&mut rng)
            .then(ActivationLayer::<12, _>::new(Tanh))
            .then(DenseLayer::<12, 1>::xavier(&mut rng));
        let mut trainer = Trainer::new(MeanSquaredError, F::build(&net, rate), 8);
        trainer.fit(&mut net, &train, 150);
        trainer.evaluate(&net, &test)
    }
}

/// Eine Zeile der Tabelle: eine Sorte, alle Lernraten, gemittelt über die Zufallsstarts.
fn row<T: Task, F: Family>() {
    let means = RATES.map(|rate| {
        let sum: f32 = (0..SEEDS).map(|seed| T::run::<F>(seed + 1, rate)).sum();
        sum / SEEDS as f32
    });
    let usable = |mean: f32| mean.is_finite() && mean <= 100.0;
    let best = means
        .into_iter()
        .filter(|&mean| usable(mean))
        .fold(f32::INFINITY, f32::min);

    print!("  {:9}", F::NAME);
    for mean in means {
        if !usable(mean) {
            print!(" {:>8} ", "div.");
        } else if mean == best {
            print!(" {mean:>8.4}*");
        } else {
            print!(" {mean:>8.4} ");
        }
    }
    println!();
}

/// Die Tabelle einer Aufgabe: ein Kopf und eine Zeile pro Sorte.
fn table<T: Task>() {
    println!("{}: {}", T::NAME, T::DETAILS);
    print!("  {:9}", "Lernrate");
    for rate in RATES {
        print!(" {rate:>8} ");
    }
    println!();
    row::<T, SgdKind>();
    row::<T, MomentumKind>();
    row::<T, NesterovKind>();
    row::<T, RmsPropKind>();
    row::<T, AdaGradKind>();
    row::<T, AdamKind>();
    row::<T, AdamWKind>();
    println!();
}

/// Das Netz, an dem der Speicherbedarf gemessen wird: eine Schicht mit 10 Eingängen und 10 Neuronen.
type Probe = DenseLayer<10, 10>;

/// Wie viel Speicher der Optimierer zusätzlich zum Netz braucht (in Bytes).
fn memory<F: Family>() {
    let net = Probe::new([[0.0; 10]; 10], [0.0; 10]);
    let optimizer = F::build(&net, 0.1);
    let bytes = size_of_val(&optimizer);
    let parameter_bytes = Probe::PARAMETER_COUNT * size_of::<f32>();
    println!(
        "  {:9} {bytes:5} Bytes  = {:.1} × die Parameter",
        F::NAME,
        bytes as f32 / parameter_bytes as f32
    );
}

fn main() {
    table::<Xor>();
    table::<Blobs>();
    table::<Sine>();

    println!(
        "Speicher des Optimierers, gemessen an einem Netz mit {} Parametern ({} Bytes):",
        Probe::PARAMETER_COUNT,
        Probe::PARAMETER_COUNT * size_of::<f32>()
    );
    memory::<SgdKind>();
    memory::<MomentumKind>();
    memory::<NesterovKind>();
    memory::<RmsPropKind>();
    memory::<AdaGradKind>();
    memory::<AdamKind>();
    memory::<AdamWKind>();
}
