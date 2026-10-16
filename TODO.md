# To-Do

Alles, was wir noch umsetzen wollen oder könnten. Neue Ideen einfach unten in die passende Gruppe schreiben.

- `[ ]` offen
- `[x]` erledigt
- Das Ausrufezeichen `!` markiert, was uns besonders wichtig ist.

Der große Block "Als Nächstes", "Eigene Schichten", "Bausteine", "Praxis", "Perceptron" und "Aufräumen"
ist abgearbeitet. Was fertig ist, steht unten unter **Erledigt**, die Entscheidungen, die dabei fielen,
unter **Entscheidungen** (ausführlicher in der [README](README.md)).

## Offen

- [ ] ! **Auf einem echten Mikrocontroller ausprobieren.** Das Testprogramm in [`mcu/`](mcu/README.md)
  läuft im Emulator QEMU auf einem Cortex-M4F und einem Cortex-M3 (ohne Betriebssystem, `std` und `alloc`,
  Ergebnis bitgleich zum Rechner), aber auf echter Hardware noch nicht. Dafür fehlt ein Gerät. Was zu
  tun ist, steht in `mcu/README.md`, Abschnitt "Auf einen echten Chip übertragen". **Bitte hier
  eintragen:** Chip, Takt und die Zeile `Flash belegt ... Stack benutzt ...` aus der Ausgabe.

### Ideen, die beim Abarbeiten aufgefallen sind

- [ ] **Rust-Version festlegen** (`rust-toolchain.toml`, und `rust-version` in der `Cargo.toml`). Die CI nimmt
  das neueste `stable`, und ein Rust-Update bringt manchmal neue Clippy-Warnungen, die die CI dann ohne
  Änderung am Code rot färben (so geschehen mit `chunks_exact`, siehe unten). Mit einer festen Version
  passiert das erst, wenn man selbst aktualisiert. Dagegen spricht: Man sieht neue Warnungen später.
- [ ] **Aufmerksamkeit (Attention) und ein Transformer-Block.** `Residual`, `LayerNorm` und `Embedding`
  gibt es, `Attention` fehlt noch. Auch sie wäre eine Operation auf einer Folge fester Länge.
- [ ] **`Concat` für U-Net-artige Netze.** `Residual`, `Fork` und `Add` stehen, aber Kanäle
  zusammenzufügen (statt zu addieren) geht noch nicht.
- [ ] **GAN:** zwei Netze mit je eigenem Optimierer. Die Bausteine passen, der `Trainer` kennt aber nur
  ein Netz.
- [ ] **Reinforcement Learning (DQN):** Ringpuffer fester Größe, Verlust, der nur eine Aktion zählt,
  Parameter in ein Zielnetz kopieren (`parameters()` und `set_parameters()` gibt es dafür schon).
- [ ] **`BatchNorm` mit echter Statistik des Mini-Batches.** Jetzt rechnet sie mit laufender Statistik
  (siehe unten), weil jedes Beispiel einzeln durch das Netz geht. Echte Batch-Statistik bräuchte eine
  Operation, die den ganzen Batch sieht.
- [ ] **Variable Folgenlängen** für RNN, GRU und LSTM: ein Auffüll-Helfer samt Maske. Die Länge steht
  bisher fest im Typ.
- [ ] **Faltung für große Bilder:** `im2col` oder kachelweises Rechnen. Die Messung
  ([`bench_conv`](crates/neuronws-layers/examples/bench_conv.rs)) zeigt, dass es sich erst bei viel
  größeren Bildern und Kanalzahlen lohnt, als eine Bibliothek ohne Heap sinnvoll auf den Stack bekommt.
- [ ] **Optimierer-Vergleich mit Lernraten-Plan und längerem Training.**
  [`optimizer_comparison`](crates/neuronws-layers/examples/optimizer_comparison.rs) nimmt eine feste
  Rundenzahl ohne Plan. Mit Plan (`CosineDecay`, `Warmup`) kann es anders aussehen.
- [ ] **Der Zahlenvergleich Rechner gegen Mikrocontroller** (`scripts/mcu.sh`) prüft nur den Fehlerwert
  auf drei Stellen. Bitgleich ist er bei der Erstellung geprüft worden, steht aber nicht als Test
  im Projekt, weil er bei jeder Änderung der Rechenreihenfolge neu eingetragen werden müsste.

### Verbesserungen, bei einer Durchsicht des Codes gefunden

Nach Wichtigkeit sortiert. Jeder Punkt nennt die Stelle, an der es auffiel.

- [ ] ! **`EarlyStopping` und `NaN`.** In [`early_stopping.rs`](crates/neuronws-layers/src/early_stopping.rs)
  (`observe`) gilt ein Verlust nur als besser, wenn `validation_loss < best - min_delta`. Ist der
  allererste Prüfverlust `NaN` (Training ist explodiert), wird er als `best` gespeichert, und danach ist
  *jeder* Vergleich falsch: Es gibt nie wieder eine Verbesserung. Ein späteres `NaN` nach guten Werten
  zählt dagegen nur als "keine Verbesserung". Beides ist unschön. Vorschlag: `NaN` und `±∞` nie als
  `best` übernehmen und als eigenes Ergebnis `Progress::Diverged` melden. Dazu zuerst einen Test
  schreiben, der den Fehler zeigt.
- [ ] ! **Divergenz überall erkennen, nicht nur im Early Stopping.** Im ganzen Crate gibt es nur eine
  einzige `is_nan`-Prüfung (in `persist.rs`). `Trainer::epoch` liefert bei explodierenden Gewichten
  still `NaN` zurück, und `fit` läuft trotzdem alle Runden durch. Vorschlag: eine
  `Scalar::is_finite`-Methode (mit `x == x && x.abs() != INF` geht das ohne `std`) und eine Variante
  von `fit`, die bei nicht endlichem Verlust abbricht. Dazu passt auch ein Test, dass `ClipNorm`
  eine absichtlich zu große Lernrate abfängt.
- [ ] ! **Lizenz und Paket-Angaben fehlen.** Es gibt keine `LICENSE`-Datei, und die `Cargo.toml`-Dateien
  haben weder `license`, `description`, `repository` noch `rust-version`. Ohne Lizenz darf niemand den
  Code benutzen, auch wenn er öffentlich ist. Vorschlag: `MIT OR Apache-2.0` (wie in der
  Rust-Welt üblich) in `[workspace.package]` eintragen und beide Textdateien anlegen. Zusammen mit dem
  Punkt "Rust-Version festlegen" oben erledigen.
- [ ] **`Rng` liefert nur `f32`.** `next_f32` nutzt 24 Bit. Für `f64`-Netze (`Normal`, `Dropout`,
  `Reparameterize`) wird deshalb nur mit der Genauigkeit eines `f32` gewürfelt, und die Werte werden
  dann hochgewandelt. Für Gewichte egal, für Statistik-Tests aber ungenau. Vorschlag: `next_f64` mit 53
  Bit und generisch über `Scalar` (`rng.next_scalar::<T>()`).
- [ ] **Fehler ohne Absturz.** Über 50 `assert!` in der Bibliothek, die meisten sind echte
  Programmierfehler (Batchgröße 0, leere Daten) und das ist so entschieden. Auf einem Mikrocontroller
  bedeutet ein Absturz aber Neustart. Zumindest dort, wo *Daten* schuld sein können und nicht der
  Code (`train_test_split` mit zu wenig Beispielen, `fit` mit leeren Daten, `confusion_matrix` mit Klasse
  außerhalb des Bereichs), wären `try_`-Varianten mit einem kleinen Fehler-`enum` sauberer. Vorher
  entscheiden, wo die Grenze liegt, und in "Entscheidungen" festhalten.
- [ ] **CI ausbauen:** (a) ein Job mit der *ältesten* unterstützten Rust-Version (`rust-version`), damit
  die Zusage stimmt, (b) `cargo doc --document-private-items` einmal ansehen, (c) `cargo miri test` für
  `persist` und `values` (kein `unsafe`, aber Miri findet auch Überläufe und Indexfehler), (d) einmal
  im Monat ein geplanter Lauf mit dem neuesten `nightly`-Clippy, damit neue Warnungen früh auffallen,
  ohne die normale CI zu stören.
- [ ] **Zufallsgestützte Tests (Property-Tests) ohne Abhängigkeit.** Wir haben `Rng` schon. Ein kleiner
  Hilfsrahmen (`for seed in 0..200 { … }`) könnte für `persist` (beliebige Bytes dürfen nie abstürzen,
  nur `Err` liefern), `conv_output_size` (stimmt mit der Schleifenrechnung überein) und `SoftmaxAcross`
  (Summe ist 1, auch bei großen Eingaben) viele Fälle prüfen. Heute stehen dort einzelne Beispiele.
- [ ] **Numerische Stabilität gezielt testen.** Es gibt Tests für Softmax und Kreuzentropie, aber keinen,
  der große Werte (±1000) oder winzige Wahrscheinlichkeiten (nahe 0) durch `SoftmaxCrossEntropy`,
  `Sigmoid` und `BinaryCrossEntropy` schickt und prüft, dass nichts `NaN` oder `∞` wird.
- [ ] **Lernpfad für Einsteiger.** 18 Beispiele liegen nebeneinander, die Reihenfolge ergibt sich nur aus
  den Namen. Eine nummerierte Liste in der README ("1. `dense_forward` → 2. `learn_linear` → 3.
  `learn_xor` → …"), jeweils mit einem Satz "was man danach versteht", würde den Weg vom naiven
  Perceptron bis zum Trainer zeigen. Passt zum Ziel des Projekts, Rust und neuronale Netze zu lernen.
- [ ] **`CHANGELOG.md` und Versionsregeln.** Alles steht auf `0.1.0`. Sobald jemand anderes das nutzt,
  braucht er eine Liste der Änderungen, vor allem bei Änderungen an `Operation` und `Trainable`.
- [ ] **Die über 60 einzelnen Testdateien** in `crates/neuronws-layers/tests/` (`loss.rs`,
  `more_losses.rs`, `activation.rs`, `activation_backward.rs`, `smooth_activations.rs`,
  `more_activations.rs`, …) sind historisch gewachsen. Nach Thema zusammenlegen (ein Ordner je Thema
  mit `mod.rs`) würde das Suchen erleichtern. Reine Aufräumarbeit, kein Zeitdruck.

## Andere Netzarten

Unser Stand war ein Multi-Layer-Perceptron, der Unterbau ist aber allgemeiner: eine Kette von
Rechenschritten (`Operation`) mit handgeschriebener Rückwärtsrechnung. Die Tabelle zeigt, was daraus
geworden ist.

| Netzart | Stand | Bemerkung |
|---|---|---|
| Autoencoder | geht | Das Ziel ist die Eingabe. Für Bilder gibt es `ConvTranspose2d` zum Hochskalieren. |
| RBF-Netze | geht | Nur eine neue `Operation` (Abstand zu Zentren), nicht gebaut. |
| CNN (Faltung, Pooling) | **gebaut** | `Conv2d` (Schrittweite, Rand, Dilation), `Conv1d`, `ConvTranspose2d`, `MaxPool2d`, `AvgPool2d`. Beispiel `cnn_lines`. |
| RNN / LSTM / GRU | **gebaut** | Die ganze Folge als eine Operation mit fester Länge (`[[T; D]; STEPS]`), die Rückwärtsrechnung über die Zeit steckt in `backward`. Der Entwurf trägt. Beispiel `sequence_memory`. |
| Transformer / Attention | mit Kniff | `Residual`, `LayerNorm` und `Embedding` stehen, `Attention` fehlt. |
| ResNet, U-Net | **ResNet gebaut** | `Residual`, `Fork`, `Add`. Für ein U-Net fehlt `Concat`. |
| Dropout, VAE | **gebaut** | `Operation::forward_train` gibt dem Lern-Durchgang Zufall und Zustand. `Dropout`, `GaussianNoise`, `Reparameterize`. |
| GAN | Trainingsschleife neu | Zwei Netze mit je eigenem Optimierer. Der `Trainer` kann es nicht. |
| Reinforcement Learning (DQN) | fast | Ringpuffer fester Größe, Loss der nur eine Aktion zählt, Parameter in ein Zielnetz kopieren. |
| SOM, Hopfield, Boltzmann-Maschinen | andere Welt | Lernen ohne Gradienten. `Loss`, `backward`, `Optimizer` passen nicht. Die getrennten Capability-Traits erlauben es trotzdem: nur `Layer` und `ForwardPath` plus ein eigenes Lern-Trait. |

## Entscheidungen

Was beim Abarbeiten bewusst so entschieden wurde (die Gründe stehen in der README, Abschnitt
"Entscheidungen"):

- **Keine dynamische Variante hinter `alloc`.** `Values` ist `Copy` und besteht aus Arrays fester Größe,
  darauf bauen die Zwischenspeicher, das Gedächtnis der Optimierer und `parameters()`. Für das
  eigentliche Problem (große Netze auf dem Stack) gibt es `Box<M>` hinter `alloc`.
- **`Optimizer::step` hat keinen Fehlertyp.** Eine Batchgröße 0 ist ein Programmierfehler und bricht mit
  einer Meldung ab, statt alle Parameter still zu `NaN` zu machen.
- **`Trainable::PARAMETER_COUNT` folgt aus `Shape::LEN`** (Standardwert des Traits). Beide Angaben
  bleiben, aber sie können nicht auseinanderlaufen.
- **`libm` bleibt ein Feature.** Es darf überall benutzt werden und ist die einzige erlaubte
  Abhängigkeit, aber ohne das Feature baut die Bibliothek ohne jede Abhängigkeit. Sind `std` und
  `libm` an, gewinnt `libm`, damit jede Plattform dasselbe rechnet.
- **Die Faltung braucht 11 Zahlen im Typ.** Ein Trait `Geometry` würde nicht helfen, denn `[T; G::H]` ist
  als Array-Länge auf stabilem Rust nicht erlaubt. Stattdessen nehmen die Makros `conv2d!`, `conv1d!`,
  `conv_transpose2d!` und die `const fn conv_output_size` die Rechnerei ab.
- **Größenfehler erscheinen bei `cargo check` nur mit `check_sizes!`.** Ohne das Makro erst bei
  `cargo build`.
- **`BatchNorm` rechnet mit laufender Statistik**, nicht mit der des Mini-Batches. Diese Statistik und
  die von `Standardize` gehören nicht zu `parameters()` und werden getrennt gesichert (`encode_slice`).
- **Sprache:** Erklärungen, Kommentare, Dokumentation und Fehlermeldungen Deutsch, Bezeichner im Code und
  Commit-Nachrichten Englisch.

## Erledigt

### Dieser Durchgang

- [x] **README und Dokumentation:** [`README.md`](README.md) (Crates, Bausteine, Features, Größen im Typ,
  Stack, Speichern, Stolpersteine, Entscheidungen), Überblick im Crate selbst (`lib.rs`),
  [`mcu/README.md`](mcu/README.md). Ein Test hält das Einstiegsbeispiel der README und ihre Verweise aktuell.
- [x] **CI mit GitHub Actions** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): jede
  Feature-Kombination als eigener Job, Clippy mit `-D warnings`, Format, Dokumentation ohne Warnungen
  (auch je Kombination: Verweise auf Typen, die es ohne Feature nicht gibt, waren ein echter Fehler),
  Bare-Metal-Builds für vier Prozessoren, alle Beispiele, Test im Mikrocontroller-Emulator. Lokal
  dasselbe: `scripts/check.sh`. **Auf GitHub geprüft:** Der erste Lauf war rot, weil Clippy in Rust 1.99
  `chunks_exact` mit fester Größe in der Faltung beanstandet (jetzt `as_chunks`). Der zweite Lauf war
  grün, alle 14 Jobs.
- [x] **Trainings- und Testdaten trennen, Early Stopping:** `data::train_test_split`,
  `shuffled_train_test_split`, `EarlyStopping`, `Trainer::fit_validated`.
- [x] **Daten pro Runde mischen:** `Trainer::epoch_shuffled`, `fit_shuffled`.
- [x] **Trefferquote:** `metrics::accuracy`, `binary_accuracy`, `argmax`, `confusion_matrix`.
- [x] **Datensätze:** `datasets` (Logikgatter, Punktwolken, Linienbilder) für Beispiele und Tests. Die
  doppelte Datenerzeugung von `cnn_lines` ist weg.

Schichten:

- [x] Dropout (mit `Operation::forward_train`), `GaussianNoise`, `Reparameterize`
- [x] Residual-Verbindung (`Residual`, `Fork`, `Add`)
- [x] Diagonal-Schicht
- [x] Batch-Normalisierung und Layer-Normalisierung
- [x] Embedding, RNN, GRU, LSTM, `LastStep`, `MeanOverSteps`
- [x] Faltung erweitert: Schrittweite, Rand, Dilation, `Conv1d`, `ConvTranspose2d`
- [x] `Softmax` und `Standardize` über eine Achse (`SoftmaxAcross`, `StandardizeChannels`)
- [x] `LinearRows` (Dense für Matrizen), `Each` (eine parameterlose Operation auf jede Zeile)
- [x] `neurons(...)` funktioniert für jede Form, also auch für Bilder (Faltung und Aktivierung in einem
  Aufruf)
- [x] Weg für Zufall in `forward`: `Operation::forward_train`

Bausteine:

- [x] Aktivierungen: Leaky ReLU, ELU, GELU, Swish, SiLU, Softplus, Hard Tanh, Hard Sigmoid
- [x] Fehlerfunktionen: Huber, Mean Absolute Error, `SoftmaxCrossEntropy` (Softmax und Fehler in einem Schritt)
- [x] Startwerte: Normalverteilung (Box-Muller, `Normal`, `XavierNormal`, `HeNormal`), `Orthogonal`
- [x] Optimierer: Nesterov-Momentum, `RmsProp`, `AdaGrad`
- [x] Gradient Clipping (`ClipValue`, `ClipNorm`) und L2-Regularisierung (`L2`) als Hüllen um jeden Optimierer
- [x] Lernraten-Pläne: `CosineDecay`, `Warmup`, `ReduceOnPlateau`

Praxis:

- [x] **Gewichte speichern und laden** ohne `std` und `alloc`: Bytes mit Prüfsumme (`save_parameters`,
  `load_parameters`) und Rust-Konstanten (`write_rust_const`).
- [x] **Große Netze und Stack:** gemessen (`stack_budget`), Tabelle in der README und in `boxed`. Lösung:
  `Box<M>` hinter `alloc`, ohne `alloc` die Größe vorher kennen (`size_of`).
- [x] **Geschwindigkeit der Faltung:** gemessen (`bench_conv`) und vorwärts drei- bis siebenmal schneller
  gemacht. Weiter kommt man erst mit `im2col`, siehe oben.
- [x] **Ziffernerkennung mit Adam** (`adam_vs_sgd`): Ergebnis: Die Lernrate war der Engpass, nicht der
  Optimierer.
- [x] **Benchmarks** SGD gegen Momentum gegen Adam und weitere auf mehreren Aufgaben
  (`optimizer_comparison`).
- [x] Mikrocontroller: Programm und Emulator-Test (Cortex-M4F und Cortex-M3) sind da, die Bibliothek wird
  außerdem für Cortex-M0/M0+ und RISC-V gebaut. Die echte Hardware steht noch offen (siehe oben).

Perceptron:

- [x] Lernregel (die klassische, nicht die per Gradient) dokumentiert und gezeigt, dass sie ein
  SGD-Schritt mit `y − t` als Ausgangs-Gradient ist (Test, Bit für Bit gleich).
- [x] Beispiel `learn_gates`: AND, OR, NAND, die Grenze bei XOR und zwei Schichten als Ausweg.
- [x] `Perceptron::new` ist wieder eine `const fn` (das Perceptron hält seine Zahlen selbst, ohne `Block`).

Aufräumen:

- [x] `PARAMETER_COUNT` und `Shape::LEN`: kein Widerspruch mehr möglich (siehe Entscheidungen)
- [x] Reste des alten `Data`-Traits: nur noch ein Test-Alias, umbenannt
- [x] Namen und Doku einheitlich: Sprachregel festgelegt, englische Prosa im Code gesucht und behoben
- [x] `Optimizer::step` und `batch_size == 0`: Entscheidung, kein Fehlertyp
- [x] Stolperstein `Sgd::new(0.1)` mit `f64`-Literal dokumentiert, `Sgd::for_model` als Ausweg

Größen und Fehler:

- [x] Die Faltung hat 11 Konstanten: Makros und `conv_output_size` statt Typ-Bündel (siehe Entscheidungen)
- [x] Fehler in Größenprüfungen: `check_sizes!` und Erklärung in README und Doku
- [x] Feste Größen: dynamische Variante geprüft und verworfen (siehe Entscheidungen)

### Vorher

- [x] Capability-Traits (`Layer`, `ForwardPath`, `BackwardPath`, `Trainable`, …)
- [x] `Chain` und `.then(...)`
- [x] Aktivierungen: Identity, Step, ReLU, Sigmoid, Tanh, Softmax
- [x] Fehlerfunktionen: MSE, Binäre Kreuzentropie, Kategoriale Kreuzentropie
- [x] Mini-Batches und `train_epoch`
- [x] `Scalar` für `f32`/`f64`, `std` und `libm` (`libm` gewinnt bei beidem)
- [x] Standardisierung der Eingaben (`Standardize`)
- [x] Ziffernerkennung (`neuronws-digits`)
- [x] Schichten aus austauschbaren Teilen: `Operation`, `Block`, `Values`, `Initializer`, `neurons`, Gradienten-Check
- [x] Optimierer mit Gedächtnis in Form des Netzes (`Trainable::Shape`), kein handgezähltes `P` mehr
- [x] Momentum, Adam, AdamW
- [x] `Trainer` (Loss + Optimizer + Batchgröße) und Lernraten-Pläne (`Constant`, `StepDecay`, `LinearDecay`)
- [x] `Conv2d`, `MaxPool2d`, `AvgPool2d` mit Gradienten-Check und Training im Test
- [x] Beliebige Formen: `Values` für beliebig tief verschachtelte Arrays, `Loss`/`Trainer` für jede Form, `Elementwise` für jede Form, `Reshape` zum Umordnen (z. B. Bild zu Vektor)
- [x] Formen verallgemeinern: `Trainer`, `train_epoch` und `Loss` arbeiten mit beliebigen Formen (`Values`)
- [x] CNN: `Conv2d`, `MaxPool2d`, `AvgPool2d`, Beispiel `cnn_lines` erkennt Linien mit 91 Parametern
