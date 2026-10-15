# To-Do

Alles, was wir noch umsetzen wollen oder könnten. Neue Ideen einfach unten in die passende Gruppe schreiben.

- `[ ]` offen
- `[x]` erledigt
- Das Ausrufezeichen `!` markiert, was uns besonders wichtig ist.

## Als Nächstes

- [ ] ! **README und Dokumentation** (von dir auf später verschoben, weil sich noch Dinge ändern).
  Inhalt: Überblick über die Crates, die Bausteine (`Operation`, `Block`, `Chain`, `Trainer`), Features (`std`, `alloc`, `libm`), ein Einstiegsbeispiel.
- [ ] **CI mit GitHub Actions**: Tests mit allen Feature-Kombinationen (keine / `std` / `libm` / beide), Clippy mit `-D warnings`, Bare-Metal-Build (`thumbv7em-none-eabihf`), Beispiele ausführen.
- [ ] **Trainings- und Testdaten trennen** plus **Early Stopping** (abbrechen, wenn der Test-Verlust wieder steigt). Im Beispiel `adam_classifier` sieht man schon leichtes Overfitting.
- [ ] **Daten pro Runde mischen** im `Trainer` (z. B. `epoch_shuffled(model, data, rng)`).
- [ ] **Hilfsfunktion für die Trefferquote** (Accuracy), damit nicht jedes Beispiel `argmax` selbst schreibt.

## Eigene Schichten

- [ ] Dropout (braucht Zufall und einen Trainings-/Test-Modus)
- [ ] Residual-Verbindung (`y = x + f(x)`)
- [ ] Diagonal-Schicht (ein Gewicht pro Eingang)
- [ ] Batch-Normalisierung und Layer-Normalisierung
- [ ] Faltungsschicht (Convolution) und Pooling
- [ ] Einbettungen (Embedding) und Rekurrente Zellen (RNN/GRU/LSTM), falls wir je Folgen verarbeiten wollen

## Andere Netzarten

Unser Stand ist ein Multi-Layer-Perceptron, der Unterbau ist aber allgemeiner: eine Kette von Rechenschritten (`Operation`) mit handgeschriebener Rückwärtsrechnung. Die Einschätzungen stammen aus dem Entwurf, nichts davon ist ausprobiert.

| Netzart | Passt? | Was nötig wäre |
|---|---|---|
| Autoencoder | ja | Nichts, das Ziel ist die Eingabe. |
| RBF-Netze | ja | Nur eine neue `Operation` (Abstand zu Zentren). |
| CNN (Faltung, Pooling) | fast | Neue Operationen. `Values` kann bisher nur Vektoren und Matrizen, für Bilder mit Kanälen braucht es tiefere Verschachtelung. Wegen fester Größen muss die Ausgabegröße als eigene Zahl angegeben werden. |
| RNN / LSTM / GRU | mit Kniff | Die ganze Folge als eine Operation mit fester Länge (`[[T; D]; STEPS]`), die Rückwärtsrechnung über die Zeit steckt in `backward`. Variable Längen ohne `alloc` nur mit Auffüllen. |
| Transformer / Attention | mit Kniff | Auch eine Operation auf einer Folge, dazu Residual-Verbindungen und Layer-Norm. |
| ResNet, U-Net | Umbau am Rand | `Chain` kann nur geradeaus. Kombinatoren wie `Residual` und `Fork` nötig, der Kern bleibt unberührt. |
| Dropout, VAE | kleiner Umbau | `forward` bekommt keinen Zufall. Weg finden, einen `Rng` oder Zustand hineinzugeben. |
| GAN | Trainingsschleife neu | Zwei Netze mit je eigenem Optimierer (passt zu Option A), aber der `Trainer` kann es nicht. |
| Reinforcement Learning (DQN) | fast | Ringpuffer fester Größe, Loss der nur eine Aktion zählt, Parameter in ein Zielnetz kopieren. |
| SOM, Hopfield, Boltzmann-Maschinen | andere Welt | Lernen ohne Gradienten. `Loss`, `backward`, `Optimizer` passen nicht. Die getrennten Capability-Traits erlauben es trotzdem: nur `Layer` und `ForwardPath` plus ein eigenes Lern-Trait. |

Die drei echten Engstellen:

- [x] **Formen verallgemeinern:** `Trainer`, `train_epoch` und `Loss` arbeiten mit beliebigen Formen (`Values`). Ein allgemeiner `impl` für `[V; N]` ging doch, ganz ohne Makro.
- [ ] `Softmax` und `Standardize` kennen nur Vektoren. Für Bilder auf eine Achse (z. B. Softmax über die Kanäle) verallgemeinern.
- [ ] `Linear` (Dense) nimmt nur Vektoren. Mit `Reshape` davor und danach ist das kein Hindernis, eine Variante für Matrizen wäre aber bequemer.
- [ ] **Feste Größen:** Später eventuell eine dynamische Variante hinter dem Feature `alloc`.
- [ ] Kombinatoren für Verzweigungen: `Residual`, `Fork` (Gradienten der Zweige addieren).
- [ ] Weg für Zufall in `forward` (Dropout, VAE).

Empfohlene Reihenfolge, falls wir eine andere Netzart bauen wollen:

- [ ] ! Als Nächstes ein **CNN** (Unterbau steht: Formen sind verallgemeinert; fehlen noch Faltungs- und Pooling-Operation) (zeigt schnell, ob der Entwurf hält), danach ein **RNN** (zeigt, ob „Folge als eine Operation" trägt).

## Bausteine zum Austauschen

Aktivierungsfunktionen:
- [ ] Leaky ReLU, ELU, GELU, Swish/SiLU, Softplus
- [ ] Hartes Tanh / Hard Sigmoid (ohne `exp`, auch ohne `libm` nutzbar)

Fehlerfunktionen:
- [ ] Huber-Loss, Mean Absolute Error
- [ ] Kreuzentropie, die Softmax und Loss in einem Schritt rechnet (numerisch stabiler)

Startwerte (Initializer):
- [ ] Normalverteilung statt Gleichverteilung (Box-Muller) für Xavier/He
- [ ] Orthogonale Initialisierung

Optimierer:
- [ ] Nesterov-Momentum, RMSprop, AdaGrad
- [ ] Gradient Clipping (Gradienten begrenzen)
- [ ] L2-Regularisierung als eigener Baustein

Lernraten-Pläne:
- [ ] Cosine Decay (braucht `cos`, nur mit `std`/`libm`)
- [ ] Warmup (am Anfang langsam hochfahren)
- [ ] Plan, der auf den Verlust reagiert (Reduce on Plateau)

## Praxis

- [ ] **Gewichte speichern und laden**, ohne `std` und `alloc` (z. B. als Rust-Konstanten ausgeben oder in ein Byte-Array schreiben)
- [ ] **Große Netze**: `Block` und der Optimierer kopieren ganze Arrays (`Values: Copy`). Prüfen, ab welcher Größe der Stack Probleme macht, und ob wir auf Referenzen/Slices umstellen müssen.
- [ ] Beispiel **Ziffernerkennung** mit Adam und Softmax neu aufsetzen und mit dem alten Ergebnis vergleichen
- [ ] Beispiel auf einem echten Mikrocontroller (`no_std`) ausprobieren
- [ ] Benchmarks: SGD gegen Momentum gegen Adam auf mehreren Aufgaben

## Perceptron-Crate

- [ ] Perceptron-Lernregel (die klassische, nicht die per Gradient) sauber dokumentieren
- [ ] Beispiele für AND/OR/XOR mit dem Perceptron und die Grenzen von XOR erklären
- [ ] Prüfen, ob `Perceptron::new` wieder `const fn` werden kann (ging vor dem `Block`-Umbau)

## Aufräumen

- [ ] `Trainable::PARAMETER_COUNT` und `Shape::LEN` sind jetzt doppelt vorhanden. Prüfen, ob eine der beiden Angaben reicht.
- [ ] Reste des alten `Data`-Traits entfernen (von dir bereits als „kann weg" freigegeben)
- [ ] Namen und Doku über alle Crates einheitlich (Deutsch/Englisch) machen
- [ ] Prüfen, ob `Optimizer::step` einen eigenen Fehlertyp braucht (z. B. bei `batch_size == 0`)
- [ ] Stolperstein dokumentieren: Bei Literalen wie `Sgd::new(0.1)` kann Rust `f64` statt `f32` wählen, wenn es nichts anderes ableiten kann

## Erledigt

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
- [x] Beliebige Formen: `Values` für beliebig tief verschachtelte Arrays, `Loss`/`Trainer` für jede Form, `Elementwise` für jede Form, `Reshape` zum Umordnen (z. B. Bild zu Vektor)
