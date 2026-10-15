# neuronws auf einem Mikrocontroller

Dieses Programm lässt die Bibliothek auf einem **Cortex-M4F** laufen: ohne Betriebssystem, ohne `std`,
ohne `alloc`, als echtes Bare-Metal-Programm für das Ziel `thumbv7em-none-eabihf`. Es

1. lernt XOR (Netz mit 17 Parametern, Adam, 300 Runden),
2. speichert die Gewichte in einen Puffer auf dem Stack, lädt sie in ein zweites Netz und prüft, dass
   zu kurze und beschädigte Daten abgelehnt werden,
3. legt ein Perceptron zur Compile-Zeit an (`const`) und lässt eines ODER lernen und an XOR scheitern,
4. misst, wie viel Flash und Stack das gekostet hat.

## Was das beweist, und was nicht

Es läuft im **Emulator QEMU** (Maschine `mps2-an386`, ein Cortex-M4 mit Fließkommaeinheit), nicht auf
einem echten Chip. Der Emulator führt dieselben Maschinenbefehle aus wie der Chip, also gilt:

| Gezeigt | Nicht gezeigt |
|---|---|
| Das Programm braucht wirklich kein Betriebssystem, keinen Heap und keine `std`: Es startet aus der Vektortabelle und läuft. | Zeitverhalten: QEMU sagt nichts über Rechengeschwindigkeit oder Stromverbrauch. |
| Die Fließkommabefehle des Cortex-M4F reichen aus. | Die Grenzen eines echten Chips: Flash- und RAM-Größe, Wartezyklen, Takt. |
| Das Ergebnis ist **bitgleich** zum Rechner: Fehler vorher 0.831, nachher 0.001, auch alle Bits der vier Ausgaben (`libm` rechnet überall mit demselben Code). | Peripherie (serielle Schnittstelle, Sensoren, Timer). |

Das Beispiel auf echter Hardware auszuprobieren ist deshalb weiter offen, siehe `TODO.md`. Was dafür zu
tun ist, steht unten.

## Starten

```text
scripts/mcu.sh
```

Das Skript baut das Programm, startet es in QEMU und prüft den Exit-Code, die Meldung `ALLES OK` und
den Vergleich mit dem Rechner. QEMU muss installiert sein (Ubuntu/Debian: `sudo apt-get install
qemu-system-arm`). Von Hand:

```text
cd mcu
cargo build --release
qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -nographic \
    -semihosting-config enable=on,target=native \
    -kernel target/thumbv7em-none-eabihf/release/neuronws-mcu
```

Die Ausgabe sieht so aus (Zahlen von der Erstellung dieses Textes):

```text
neuronws auf einem Cortex-M4F: ohne Betriebssystem, ohne std, ohne alloc
Größen: Netz 264 Bytes mit 17 Parametern, Optimierer 168 Bytes
Fehler vorher 0.831, nachher 0.001
  [ok] Das Netz hat XOR gelernt (Fehler unter 0.05)
     [0.0, 0.0] -> 0.000 (soll 0, Bits 0x38da48a6)
  ...
Flash belegt: 60464 Bytes, Stack benutzt: 3696 Bytes
ALLES OK
```

Der **Flash** (60 KB) ist größer, als das Netz vermuten lässt, weil das Programm auch `libm` (tanh, exp)
und das Formatieren von Kommazahlen für die Ausgabe enthält. Der **Stack** (3,7 KB) gilt für dieses
kleine Netz im Release-Build. Für ein eigenes Netz selbst messen, mit den Funktionen `paint_stack` und
`stack_used` in `src/main.rs` (der Debug-Build braucht das Vier- bis Sechsfache), und `size_of` für Netz
und Optimierer prüfen, siehe die Tabelle in `neuronws_layers::boxed`.

## Wie das Programm aufgebaut ist

| Datei | Aufgabe |
|---|---|
| `Cargo.toml` | Eigener Workspace, weil das Programm `unsafe` braucht (im Hauptprojekt verboten). Abhängigkeiten: `neuronws-layers` (ohne `std`, ohne `alloc`, mit `libm`) und `neuronws-perceptrons`. |
| `build.rs` | Sagt dem Linker, wo `link.ld` liegt. |
| `link.ld` | Der Speicherplan: Vektortabelle ganz vorn im Flash, dann Code und Konstanten. Bricht das Linken ab, falls das Programm veränderliche Statics hätte, denn der Startcode kopiert kein `.data` und nullt kein `.bss`. |
| `src/main.rs` | Startcode (Vektortabelle, Reset, Fließkommaeinheit einschalten), Ausgabe und Ende über Semihosting, Messung von Stack und Flash und der eigentliche Test. |

Den Startcode liefert sonst die Bibliothek `cortex-m-rt`. Er steht hier selbst im Programm, damit das
Projekt ohne fremde Bibliotheken auskommt (bis auf `libm`). Es sind gut 30 Zeilen.

**Semihosting** ist der Weg, über den das Programm mit der Außenwelt spricht: Es führt den Befehl
`bkpt 0xAB` aus, und der Debugger (hier QEMU) erkennt ihn als Bitte, einen Text auszugeben oder das
Programm zu beenden. Das braucht keine Treiber und keine serielle Schnittstelle.

## Auf einen echten Chip übertragen

1. **Ziel wählen.** Cortex-M4F und M7 mit Fließkommaeinheit: `thumbv7em-none-eabihf` (wie hier).
   Cortex-M3 ohne Fließkommaeinheit: `thumbv7m-none-eabi`; die Bibliothek rechnet dort in Software,
   das ist langsamer, geht aber ohne Änderung am Code.
2. **Speicherplan** in `link.ld` anpassen: `ORIGIN` und `LENGTH` von `FLASH` und `RAM` stehen im
   Datenblatt des Chips.
3. **Ausgabe ersetzen.** Semihosting funktioniert nur mit angeschlossenem Debugger (zum Beispiel
   `probe-rs` oder OpenOCD mit `monitor arm semihosting enable`). Ohne Debugger bleibt der
   `bkpt`-Befehl hängen. Im Betrieb wird `Line::flush` durch das Schreiben auf eine serielle
   Schnittstelle ersetzt und `exit` durch eine Endlosschleife oder einen Neustart.
4. **Takt und Flash-Wartezyklen** stellen die meisten Chips vor dem Rechnen ein (das Hersteller-Paket
   oder ein PAC hilft dabei). Der Reset-Code in `src/main.rs` schaltet nur die Fließkommaeinheit ein.
5. **Flashen**, zum Beispiel mit `probe-rs run --chip <CHIP> target/thumbv7em-none-eabihf/release/neuronws-mcu`.

Wer das ausprobiert: Das wäre der Nachweis, der hier fehlt. Bitte Chip, Takt und die Zeile
`Flash belegt ... Stack benutzt ...` in `TODO.md` eintragen.
