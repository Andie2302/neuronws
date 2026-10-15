# neuronws

Neuronale Netze in Rust, die **ohne Betriebssystem** laufen: `no_std`, ohne `alloc`, ohne `std` und ohne
fremde Bibliotheken. Die einzige erlaubte Abhängigkeit ist [`libm`](https://crates.io/crates/libm) für
Mathe-Funktionen wie `exp` und `tanh`, und auch die nur hinter einem Feature. Alle Größen eines Netzes
stehen im Typ, das Netz liegt komplett auf dem Stack (ein fertiges, nur benutztes Netz auch als `const`
oder `static` im Flash).

Das Projekt ist zum Lernen gebaut: Die Rückwärtsrechnung jeder Schicht ist von Hand geschrieben und
gegen die numerische Ableitung geprüft, und die Dokumentation erklärt, **warum** etwas so gemacht ist.

## Die Crates

| Ordner | Inhalt |
|---|---|
| [`crates/neuronws-layers`](crates/neuronws-layers) | Die Bibliothek: Schichten, Aktivierungen, Fehlerfunktionen, Optimierer, Training, Datensätze, Gewichte speichern. |
| [`crates/neuronws-perceptrons`](crates/neuronws-perceptrons) | Das klassische Perceptron mit Rosenblatts Lernregel. Eine `const fn`, ohne Gradienten. |
| [`crates/neuronws-digits`](crates/neuronws-digits) | Ziffernerkennung auf 5×7-Bildern: Daten, Rauschen, ein passendes Netz und Beispiele. |
| [`mcu`](mcu) | Ein Testprogramm für einen Cortex-M4F, das im Emulator QEMU läuft (eigener Workspace, siehe unten). |

## Einstieg

Ein kleines Netz lernt XOR. Das ist das Beispiel [`quickstart`](crates/neuronws-layers/examples/quickstart.rs):

```rust
//! Einstieg: Ein kleines Netz lernt XOR.
//!
//! Start: `cargo run -p neuronws-layers --features libm --example quickstart`

use neuronws_layers::activation::{Sigmoid, Tanh};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::adam::Adam;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::datasets::logic;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::BinaryCrossEntropy;
use neuronws_layers::random::Rng;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::ForwardPath;

fn main() {
    let mut rng = Rng::new(1);

    // 1. Das Netz: 2 Eingänge → 4 verborgene Neuronen (Tanh) → 1 Ausgang (Sigmoid).
    //    Alle Größen stehen im Typ, nichts davon braucht den Heap.
    let mut net = DenseLayer::<2, 4>::xavier(&mut rng)
        .then(ActivationLayer::<4, _>::new(Tanh))
        .then(DenseLayer::<4, 1>::xavier(&mut rng))
        .then(ActivationLayer::<1, _>::new(Sigmoid));

    // 2. Wie gelernt wird: Fehlerfunktion, Optimierer und Batchgröße.
    let data = logic::as_vectors(&logic::XOR);
    let mut trainer = Trainer::new(BinaryCrossEntropy, Adam::for_model(&net, 0.1), 4);

    // 3. Lernen: 300 Runden über alle vier Beispiele.
    let before = trainer.evaluate(&net, &data);
    let after = trainer.fit(&mut net, &data, 300);
    println!("Fehler vorher {before:.3}, nachher {after:.3}");

    // 4. Benutzen.
    for (input, target) in &data {
        let output = net.forward(input)[0];
        println!("{input:?} -> {output:.2} (soll {})", target[0]);
    }
}
```

Start: `cargo run -p neuronws-layers --features libm --example quickstart`. Die Ausgabe:

```text
Fehler vorher 0.831, nachher 0.001
[0.0, 0.0] -> 0.00 (soll 0)
[0.0, 1.0] -> 1.00 (soll 1)
[1.0, 0.0] -> 1.00 (soll 1)
[1.0, 1.0] -> 0.00 (soll 0)
```

Weiter geht es mit den Beispielen in [`crates/neuronws-layers/examples`](crates/neuronws-layers/examples),
grob in dieser Reihenfolge: `dense_forward`, `learn_linear`, `learn_xor`, `learn_with_trainer`,
`adam_classifier`, `cnn_lines`, `build_your_own_layer`. Alle Beispiele laufen mit
`scripts/examples.sh`.

## Die Bausteine

Vier Ideen tragen alles:

1. **`Operation`** ist die reine Mathematik einer Schicht: `forward` (wie wird aus der Eingabe die
   Ausgabe?) und `backward` (wie fließt der Gradient zurück?). Mehr muss man für eine neue Schicht nicht
   schreiben. [`build_your_own_layer`](crates/neuronws-layers/examples/build_your_own_layer.rs) zeigt
   es.
2. **`Block<Op>`** macht aus jeder Operation eine Schicht: Er hält die Parameter, ihre Gradienten und
   die Zwischenwerte, die die Rückwärtsrechnung braucht. Damit bekommt jede Operation alle Fähigkeiten
   (siehe Tabelle).
3. **`Chain`** und `.then(...)` verketten Schichten. Ob Ausgabe und nächste Eingabe zusammenpassen,
   prüft der Compiler an den Typen.
4. **`Trainer`** bündelt, *wie* gelernt wird: Fehlerfunktion (`Loss`), Optimierer (`Optimizer`) und
   Batchgröße. Das Netz kennt keine Lernregel, der Trainer kennt keine Schichten.

Die Fähigkeiten einer Schicht sind getrennte Traits, damit jede Schicht nur verspricht, was sie kann:

| Trait | Fähigkeit |
|---|---|
| `Layer` | Marke: Dies ist eine Schicht. |
| `ForwardPath` | Vorwärts rechnen im **Einsatz** (`&self`, ohne Zufall: Dropout ist aus). |
| `ForwardPathMut` | Vorwärts rechnen zum **Lernen**: Das Netz merkt sich Zwischenwerte, Dropout ist an. |
| `BackwardPathMut` | Gradienten zurückrechnen und aufsummieren. |
| `Trainable` | Parameter samt Gradienten herausgeben (für Optimierer, Speichern, Zählen). |

Eingaben, Ausgaben und Parameter sind **`Values`**: beliebig verschachtelte Arrays und Tupel, vom
Vektor `[T; N]` über die Matrix `[[T; C]; R]` bis zum Bild `[[[T; W]; H]; K]`. Alles hat eine feste
Größe, alles ist `Copy`, nichts braucht den Heap.

### Was es gibt

- **Schichten:** `DenseLayer` (und `LinearRows` für Matrizen), `Diagonal`, Faltung (`Conv2d`, `Conv1d`,
  `ConvTranspose2d`) mit Schrittweite, Rand und Dilation, `MaxPool2d`, `AvgPool2d`, `LayerNorm`,
  `BatchNorm`, `Dropout`, `GaussianNoise`, `Reparameterize`, `Residual`, `Fork`, `Add`, `Reshape`,
  `Standardize`, `Softmax` (auch über eine Achse), `Embedding`, `Rnn`, `Gru`, `Lstm` und die Folgen-Helfer
  `LastStep` und `MeanOverSteps`. Mit `neurons(...)` lassen sich Rechnung, Startwerte und Aktivierung
  einer Schicht einzeln austauschen.
- **Aktivierungen:** Identity, Step, ReLU, Leaky ReLU, ELU, GELU, Swish, SiLU, Softplus, Sigmoid, Tanh,
  Hard Tanh, Hard Sigmoid.
- **Fehlerfunktionen:** MSE, Mean Absolute Error, Huber, binäre und kategoriale Kreuzentropie und
  `SoftmaxCrossEntropy` (Softmax und Fehler in einem Schritt, numerisch stabiler).
- **Startwerte:** Zeros, Constant, Uniform, Xavier, He, Normal, XavierNormal, HeNormal, Orthogonal.
- **Optimierer:** `Sgd`, `Momentum` (mit `with_nesterov`), `RmsProp`, `AdaGrad`, `Adam`, `AdamW`. Dazu
  Hüllen, die um jeden Optimierer passen: `ClipValue`, `ClipNorm` und `L2`.
- **Lernraten-Pläne:** `Constant`, `StepDecay`, `LinearDecay`, `CosineDecay`, `Warmup` und der
  verlustabhängige `ReduceOnPlateau`.
- **Training:** `Trainer` (mit Mischen, Plänen und Prüfdaten), `train_test_split`, `EarlyStopping`,
  Metriken (`accuracy`, `binary_accuracy`, `confusion_matrix`).
- **Daten:** `datasets` mit den Logikgattern, drei Punktwolken (`blobs`) und Linienbildern (`lines`), jeweils
  deterministisch aus einem `Rng` (ein eigener Zufallsgenerator, xorshift64*, ohne Abhängigkeit).
- **Weiteres:** Gradienten-Check (`gradcheck`) für eigene Schichten, Gewichte speichern und laden
  (`persist`).

Was jeweils dahintersteckt, steht in der Dokumentation: `cargo doc --workspace --no-deps --open`.

## Features: `std`, `alloc` und `libm`

`neuronws-layers` ist **standardmäßig** `no_std`, ohne `alloc` und ohne `std` (`default = []`). Alles
Weitere ist ein Feature, und jedes Feature steht wirklich hinter seinem Schalter.

| Feature | Was es bringt |
|---|---|
| *(keins)* | Reines `core`, keine Abhängigkeit. Läuft auf jedem Ziel. |
| `libm` | Mathe-Funktionen (`exp`, `tanh`, `ln`, `sqrt`, `cos`, `sin`) aus der Bibliothek `libm`. Sie ist reines Rust und `no_std`. |
| `std` | Dieselben Mathe-Funktionen aus der Standardbibliothek. Das Crate hängt dann an `std`. |
| `alloc` | Alles, was den Heap braucht: `Box<M>` als Schicht, damit große Netze nicht auf dem Stack liegen. Braucht einen globalen Allocator, aber nicht `std`. |

Sind `std` **und** `libm` an, gewinnt `libm`: Dann rechnet jede Plattform mit demselben Code und bekommt
dieselben Ergebnisse. Das ist gemessen: Das XOR-Beispiel liefert auf dem x86-64-Rechner, dem emulierten
Cortex-M4F (Hardware-Fließkomma) und dem emulierten Cortex-M3 (Software-Fließkomma) dieselben Ausgaben,
bis auf das letzte Bit. `scripts/mcu.sh` vergleicht bei jedem Lauf die Bits der beiden Prozessoren und den
Fehlerwert mit dem Rechner (siehe [`mcu`](mcu)).

`libm` darf überall benutzt werden, es ist die einzige erlaubte Abhängigkeit. Es bleibt trotzdem ein
Feature statt einer festen Abhängigkeit, damit das Crate ohne jede Abhängigkeit baut, wenn man keine
Mathe-Funktionen braucht. Wer sie braucht, schreibt `features = ["libm"]` in die eigene `Cargo.toml`.

**Was geht ohne Feature?** Dense, Diagonal, Faltung, Pooling, `Reshape`, `Residual`, `Dropout`, `Rnn` (mit
Hard Tanh); Identity, Step, ReLU, Leaky ReLU, Hard Tanh und Hard Sigmoid; MSE, MAE und Huber; Zeros,
Constant und Uniform; `Sgd`, `Momentum`, `ClipValue` und `L2`; alle Pläne außer `CosineDecay`;
`Standardize` (mit bekanntem Mittelwert und bekannter Streuung); Training, Metriken, Datensätze,
Gewichte speichern und laden, der Gradienten-Check. Das Perceptron-Crate braucht nie ein Feature.

**Was braucht `libm` oder `std`?** Sigmoid, Tanh, ELU, GELU, Swish, SiLU, Softplus; die Kreuzentropien;
Xavier, He, Normal, XavierNormal, HeNormal, Orthogonal; `RmsProp`, `AdaGrad`, `Adam`, `AdamW`,
`ClipNorm`; `Softmax`, `LayerNorm`, `BatchNorm`; `Standardize` aus Daten berechnen; `CosineDecay`; `Gru`
und `Lstm`; `Rng::next_normal` (und damit `GaussianNoise` und `Reparameterize`).

Die Beispiele brauchen `--features libm`. Ohne das Feature meldet Cargo: *target requires the features:
libm*.

## Größen stehen im Typ, und wann Fehler auffallen

Weil kein Heap da ist, steht jede Größe im Typ: `DenseLayer<2, 4>` hat 2 Eingänge und 4 Neuronen. Das
ist die Stärke (der Compiler prüft, ob Schichten zusammenpassen) und die Grenze: Rust kann in einem Typ
noch nicht rechnen. Eine Faltung `Conv2d` braucht deshalb **11 Zahlen**, darunter die Ausgabegröße `OH`
und `OW`, die man sonst aus `(H + 2·PAD − DIL·(KH − 1) − 1) / STRIDE + 1` bekäme. Dafür bräuchte es die
instabile Sprachfunktion `generic_const_exprs`. Ein Typ, der die Zahlen als Bündel trägt (etwa ein Trait
`Geometry`), würde nicht helfen, denn auch `[T; G::H]` ist als Array-Länge auf stabilem Rust nicht
erlaubt.

Zwei Hilfen nehmen die Arbeit ab:

- Die Makros `conv2d!`, `conv2d_layer!`, `conv1d!` und `conv_transpose2d!` schreiben den Typ hin und
  rechnen `OH` und `OW` mit `conv_output_size` selbst aus:
  `conv2d!(in: 1, out: 4, size: (28, 28), kernel: (5, 5), stride: 2, padding: 2)`.
- Stimmt eine Zahl nicht, bricht das Kompilieren ab, aber **erst bei `cargo build` oder `cargo test`,
  nicht bei `cargo check`** und nicht in der Anzeige der IDE. Die Prüfung steckt in `new()` und läuft,
  wenn der Compiler den konkreten Typ einsetzt. Das Makro **`check_sizes!`** zieht sie nach vorn: Es ruft
  `new()` in einer `const`, und die wertet der Compiler immer sofort aus. Einmal an einer Stelle neben
  dem Netz aufschreiben:

  ```rust
  neuronws_layers::check_sizes!(
      Conv2dValid<1, 4, 6, 6, 3, 3, 4, 4>,
      MaxPool2d<4, 4, 4, 2, 2, 2>,
      Reshape<Image<4, 2, 2>, [f32; 16]>,
  );
  ```

  (Mit den passenden `use`-Zeilen, siehe die Dokumentation von `check_sizes!`. Die Typen brauchen eine
  `const fn new()` ohne Argumente.)

## Große Netze und der Stack

Jeder `Block` hält Eingabe, Ausgabe und Eingabe-Gradient als Arrays, dazu Parameter und Gradienten, und
der Optimierer legt sein Gedächtnis dazu. Bei Bildern kommt schnell einiges zusammen, und der Stack ist
klein (2 MB in einem Test-Thread, auf einem Mikrocontroller oft nur wenige KB). Gemessen mit
[`stack_budget`](crates/neuronws-layers/examples/stack_budget.rs), kleinster Stack, mit dem ein
Lernschritt (vorwärts, rückwärts, AdamW) noch läuft, `f32`, x86-64:

```text
                                   Größe     Stack mindestens
                                   im RAM    Release    Debug
Dense 784 → 64 → 10                 812 KB    1052 KB   4255 KB
Faltung  8 und 16 Kanäle            276 KB     591 KB   2008 KB
Faltung 16 und 32 Kanäle            583 KB    1193 KB   4116 KB
Faltung 32 und 64 Kanäle           1304 KB    2575 KB   8817 KB
Faltung 64 und 128 Kanäle          3177 KB    5852 KB  20211 KB
32/64, jede Schicht in einer Box      0 KB     517 KB   2063 KB
64/128, jede Schicht in einer Box     0 KB    1459 KB   6122 KB
```

- **Faustregel:** Im Release-Build braucht ein Lernschritt etwa doppelt so viel Stack, wie Netz und
  Optimierer groß sind. Im Debug-Build das Vier- bis Sechsfache.
- **Mit `std`** hilft ein größerer Stack sofort: `std::thread::Builder::new().stack_size(...)`.
- **Mit `alloc`** legt `Box::new(schicht)` jede Schicht auf den Heap. Dann braucht der Stack nur noch
  die größte einzelne Schicht, im Release-Build etwa ein Viertel bis ein Fünftel.
- **Ohne beides** (Mikrocontroller) bleibt, das Netz klein zu halten und seine Größe vorher zu kennen:
  `core::mem::size_of::<Netz>()` und `size_of_val(&optimizer)` sind exakt. Das XOR-Netz im Testprogramm
  braucht 3,7 KB Stack, siehe [`mcu`](mcu).

## Gewichte speichern und laden

Ohne `std` und ohne `alloc`, in feste Puffer:

- `save_parameters` und `load_parameters` schreiben die lernbaren Parameter als Bytes (Kennung, Zahlenart,
  Anzahl, Werte, CRC-32), für Flash, EEPROM, Datei oder Funk. Schlägt das Laden fehl (falsche Größe,
  beschädigte Daten), bleibt das Netz unverändert. `byte_len::<Netz>()` ist eine `const fn` für den Puffer.
- `write_rust_const` gibt die Gewichte als Text einer Rust-Konstante aus, die man fest in das Programm
  einkompiliert.

Gespeichert werden nur **gelernte** Parameter. Zustand, der nicht gelernt wird (die laufende Statistik
einer `BatchNorm`, Mittelwert und Streuung einer `Standardize`), gehört nicht dazu und wird mit
`encode_slice` getrennt gesichert.

## Auf dem Mikrocontroller

[`mcu`](mcu) ist ein echtes Bare-Metal-Programm ohne Betriebssystem, `std` und `alloc`. Es lernt XOR,
speichert und lädt Gewichte, prüft das Perceptron und misst Flash und Stack. Es läuft im Emulator QEMU
(`scripts/mcu.sh`) auf einem **Cortex-M4F** (Fließkomma in der Hardware) und auf einem **Cortex-M3**
(Fließkomma in Software) und liefert auf beiden und auf dem Rechner dieselben Zahlen, bis auf das letzte
Bit. Dazu wird die Bibliothek für den Cortex-M0/M0+ (zum Beispiel RP2040) und für RISC-V gebaut, in
jeder Feature-Kombination.

Auf **echter Hardware** ist es noch nicht ausprobiert, siehe [`mcu/README.md`](mcu/README.md) und die
offene Aufgabe in [`TODO.md`](TODO.md).

## Stolpersteine

- **`f32` oder `f64`?** Bei `Sgd::new(0.1)` leitet Rust den Zahlentyp meist aus dem Netz ab. Gibt es
  nichts, woraus es ihn ableiten kann, nimmt es für ein Kommazahl-Literal `f64`, und eine Variable hat
  ihren Typ schon. Der Fehler (*`Sgd<f64>` ist kein Optimierer für dieses Netz*) erscheint dann erst bei
  `step`, weit weg von der Ursache. `Sgd::for_model(&net, 0.1)` (und dasselbe bei `Momentum`, `Adam`,
  `RmsProp`, ...) lässt das Netz den Typ bestimmen.
- **Größenfehler erst beim Bauen:** siehe oben, `check_sizes!` hilft.
- **Der Debug-Build braucht viel mehr Stack** als der Release-Build (Faktor 4 bis 6), siehe oben. Ein
  Stack-Überlauf bricht mit `std` mit der Meldung *has overflowed its stack* ab. Auf einem
  Mikrocontroller bleibt er oft stumm und zeigt sich als Absturz oder als beschädigte Daten.
- **`forward` und `forward_mut` sind nicht dasselbe:** Nur `forward_mut` (Lernen) schaltet Dropout und
  Rauschen ein und führt Statistik nach. Zum Benutzen eines fertigen Netzes immer `forward`.
- **`BatchNorm` rechnet mit laufender Statistik**, nicht mit der Statistik des Mini-Batches (jedes Beispiel
  wird einzeln durch das Netz geschickt, ein Mini-Batch ist hier nur eine Summe von Gradienten). Die
  Statistik ist nicht in `parameters()`, siehe oben.
- **Optimierer bei gleicher Lernrate zu vergleichen sagt wenig:** Die beste Lernrate unterscheidet sich
  zwischen Optimierern um Größenordnungen. Wer mit einer Rate für alle vergleicht, vergleicht die
  Rate. Das Beispiel [`optimizer_comparison`](crates/neuronws-layers/examples/optimizer_comparison.rs)
  zeigt es.
- **GRU und LSTM schützen das Gedächtnis nur, wenn ihre Tore am Anfang offen stehen:** Bei Länge 32 lernt
  die GRU mit Bias 0 nur 54 %, mit Bias 2 dagegen 99,9 %, siehe
  [`sequence_memory`](crates/neuronws-layers/examples/sequence_memory.rs).
- **Kleine Stichproben täuschen:** Bei harten Aufgaben gelingt ein Zufallsstart ganz oder gar nicht.
  Mittel aus wenigen Starts springen deshalb in großen Schritten, und Unterschiede zwischen zwei
  Spalten sind dann oft nur ein Start mehr oder weniger.

## Entscheidungen

Was bewusst so gebaut ist, und warum:

- **Keine dynamische Variante (Größen zur Laufzeit) hinter `alloc`.** `Values` ist `Copy` und besteht
  aus Arrays fester Größe. Darauf bauen die Zwischenspeicher der Schichten, der Optimierer
  (sein Gedächtnis hat die Form des Netzes) und `parameters()`, das ein Netz als Wert kopiert. Eine
  dynamische Variante bräuchte einen zweiten, auf Slices und `Vec` gebauten Satz davon und damit jede
  Schicht doppelt. Für das eigentliche Problem, große Netze auf dem Stack, gibt es `Box<M>` hinter
  `alloc`. Größen zur Laufzeit gibt es nicht.
- **`Optimizer::step` hat keinen Fehlertyp.** Der einzige Fehler wäre `batch_size == 0`. Die Batchgröße
  kommt in der Praxis aus `chunks()` oder ist eine Konstante, eine 0 ist ein Programmierfehler. `step`
  bricht dann mit einer Meldung ab, statt alle Parameter still zu `NaN` zu machen.
- **`Trainable::PARAMETER_COUNT` folgt aus `Shape::LEN`.** Die Zahl ist ein Standardwert des Traits und kann
  deshalb nicht von der Form abweichen. Der Name bleibt, weil er in Konstanten wie `byte_len` bequemer ist.
- **Variable Folgenlängen** (RNN) gibt es ohne Heap nur mit Auffüllen. Die Länge steht im Typ.

## Prüfen

| Befehl | Was er tut |
|---|---|
| `scripts/check.sh quick` | Format, Clippy (`-D warnings`) und Tests ohne Feature und mit `libm`. Für zwischendurch. |
| `scripts/check.sh` | Dazu alle Feature-Kombinationen (auch bei der Dokumentation), Bare-Metal-Builds für vier Prozessoren, alle Beispiele und der Test im Emulator (falls `qemu-system-arm` da ist). Dasselbe prüft die CI. |
| `scripts/examples.sh` | Baut und startet alle Beispiele im Release-Build. |
| `scripts/mcu.sh` | Baut das Mikrocontroller-Programm für Cortex-M4F und Cortex-M3 und führt es in QEMU aus. |

Die CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) macht dasselbe, jede Feature-Kombination
als eigener Job. Die Bare-Metal-Builds (Cortex-M4F, M3, M0/M0+ und RISC-V) sind der eigentliche Beweis
für `no_std` ohne `alloc`: Dort gibt es keine `std` und keinen Heap, jede versehentliche Benutzung wäre
ein Fehler.

## Konventionen

- **Sprache:** Erklärungen, Kommentare, Dokumentation und Fehlermeldungen sind auf Deutsch. Bezeichner
  im Code und Commit-Nachrichten sind auf Englisch, so wie es das Rust-Umfeld erwartet.
- **`unsafe` ist in der ganzen Bibliothek verboten** (`unsafe_code = "forbid"`). Nur das Testprogramm
  in `mcu/` braucht es (Startcode, Semihosting) und ist deshalb ein eigener Workspace.
- **Keine fremden Bibliotheken**, außer `libm` hinter dem gleichnamigen Feature.
- **Jede Schicht hat einen Gradienten-Check** gegen die numerische Ableitung (`gradcheck`). Bei Schichten mit
  Zufall (`Dropout`, `GaussianNoise`, `Reparameterize`) gilt er für die gezogene Maske beziehungsweise das
  gezogene Rauschen.

Offene Aufgaben und Ideen stehen in [`TODO.md`](TODO.md).
