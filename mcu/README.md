# neuronws auf einem Mikrocontroller

Dieses Programm lässt die Bibliothek als **echtes Bare-Metal-Programm** laufen: ohne Betriebssystem,
ohne `std`, ohne `alloc`. Es läuft auf zwei Prozessoren:

| Ziel | Prozessor | Fließkomma | Emulierte Maschine |
|---|---|---|---|
| `thumbv7em-none-eabihf` | Cortex-M4F | in der Hardware (FPU) | QEMU `mps2-an386` |
| `thumbv7m-none-eabi` | Cortex-M3 | in Software (der Compiler rechnet es aus) | QEMU `mps2-an385` |

Was es tut:

1. Es lernt XOR (Netz mit 17 Parametern, Adam, 300 Runden).
2. Es speichert die Gewichte in einen Puffer auf dem Stack, lädt sie in ein zweites Netz und prüft, dass
   zu kurze und beschädigte Daten abgelehnt werden.
3. Es legt ein Perceptron zur Compile-Zeit an (`const`) und lässt eines ODER lernen und an XOR scheitern.
4. Es misst, wie viel Flash und Stack das gekostet hat.

## Was das beweist, und was nicht

Es läuft im **Emulator QEMU**, nicht auf einem echten Chip. Der Emulator führt dieselben Maschinenbefehle
aus wie der Chip, also gilt:

| Gezeigt | Nicht gezeigt |
|---|---|
| Das Programm braucht wirklich kein Betriebssystem, keinen Heap und keine `std`: Es startet aus der Vektortabelle und läuft. | Zeitverhalten: QEMU sagt nichts über Rechengeschwindigkeit oder Stromverbrauch. |
| Mit und ohne Fließkommaeinheit: Der Cortex-M4F rechnet in der Hardware, der Cortex-M3 in Software. | Die Grenzen eines echten Chips: Flash- und RAM-Größe, Wartezyklen, Takt. |
| Das Ergebnis ist **bitgleich** auf drei Rechnerarten: Fehler vorher 0.831, nachher 0.001, auch alle Bits der vier Ausgaben auf dem x86-64-Rechner, dem Cortex-M4F und dem Cortex-M3 (`libm` rechnet überall mit demselben Code). | Peripherie (serielle Schnittstelle, Sensoren, Timer). |

Das Beispiel auf echter Hardware auszuprobieren ist deshalb weiter offen, siehe `TODO.md`. Was dafür zu
tun ist, steht unten.

Zusätzlich wird die Bibliothek (nur gebaut, nicht ausgeführt) für den **Cortex-M0/M0+** (`thumbv6m-none-eabi`,
zum Beispiel der RP2040 im Raspberry Pi Pico) und für **RISC-V** (`riscv32imc-unknown-none-elf`) übersetzt,
in jeder Feature-Kombination. Beide haben weder Fließkommaeinheit noch atomare Befehle, und RISC-V
hat dazu einen ganz anderen Befehlssatz.

## Starten

```text
scripts/mcu.sh
```

Das Skript baut das Programm für beide Ziele, startet jedes in QEMU und prüft den Exit-Code, die Meldung
`ALLES OK`, den Vergleich des Fehlerwerts mit dem Rechner und dass die Ausgaben beider Prozessoren bis
auf das letzte Bit gleich sind. QEMU muss installiert sein (Ubuntu/Debian: `sudo apt-get install
qemu-system-arm`). Von Hand, hier für den Cortex-M4F:

```text
cd mcu
cargo build --release
qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -nographic \
    -semihosting-config enable=on,target=native \
    -kernel target/thumbv7em-none-eabihf/release/neuronws-mcu
```

Für den Cortex-M3: `--target thumbv7m-none-eabi` beim Bauen, `-machine mps2-an385 -cpu cortex-m3` bei
QEMU und `target/thumbv7m-none-eabi/...` als Pfad. QEMU schreibt die Ausgabe auf `stderr`.

Die Ausgabe sieht so aus (Zahlen von der Erstellung dieses Textes, Cortex-M4F):

```text
neuronws ohne Betriebssystem, ohne std, ohne alloc
Ziel: thumbv7em-none-eabihf (Cortex-M4F, Fließkomma in der Hardware)
Größen: Netz 264 Bytes mit 17 Parametern, Optimierer 168 Bytes
Fehler vorher 0.831, nachher 0.001
  [ok] Das Netz hat XOR gelernt (Fehler unter 0.05)
     [0.0, 0.0] -> 0.000 (soll 0, Bits 0x38da48a6)
  ...
Flash belegt: 60624 Bytes, Stack benutzt: 3696 Bytes
ALLES OK
```

Gemessen auf den beiden Prozessoren:

| | Flash | Stack |
|---|---|---|
| Cortex-M4F (`thumbv7em-none-eabihf`) | 60 624 Bytes | 3696 Bytes |
| Cortex-M3 (`thumbv7m-none-eabi`) | 48 736 Bytes | 3792 Bytes |

Der **Flash** (rund 50 bis 60 KB) ist größer, als das Netz vermuten lässt, weil das Programm auch `libm`
(tanh, exp), die Fließkomma-Routinen des Compilers und das Formatieren von Kommazahlen für die Ausgabe
enthält. Der **Stack** (knapp 4 KB) gilt für dieses kleine Netz im Release-Build. Für ein eigenes Netz
selbst messen, mit den Funktionen `paint_stack` und `stack_used` in `src/main.rs` (der Debug-Build braucht
das Vier- bis Sechsfache), und `size_of` für Netz und Optimierer prüfen, siehe die Tabelle in
`neuronws_layers::boxed`.

## Wie das Programm aufgebaut ist

| Datei | Aufgabe |
|---|---|
| `Cargo.toml` | Eigener Workspace, weil das Programm `unsafe` braucht (im Hauptprojekt verboten). Abhängigkeiten: `neuronws-layers` (ohne `std`, ohne `alloc`, mit `libm`) und `neuronws-perceptrons`. |
| `build.rs` | Sagt dem Linker, wo `link.ld` liegt. |
| `link.ld` | Der Speicherplan: Vektortabelle ganz vorn im Flash, dann Code und Konstanten. Bricht das Linken ab, falls das Programm veränderliche Statics hätte, denn der Startcode kopiert kein `.data` und nullt kein `.bss`. |
| `src/main.rs` | Startcode (Vektortabelle, Reset, Fließkommaeinheit einschalten), Ausgabe und Ende über Semihosting, Messung von Stack und Flash und der eigentliche Test. |

Den Startcode liefert sonst die Bibliothek `cortex-m-rt`. Er steht hier selbst im Programm, damit das
Projekt ohne fremde Bibliotheken auskommt (bis auf `libm`). Es sind gut 30 Zeilen. Die Fließkommaeinheit
wird nur eingeschaltet, wenn es eine gibt (`cfg(target_abi = "eabihf")`).

**Semihosting** ist der Weg, über den das Programm mit der Außenwelt spricht: Es führt den Befehl
`bkpt 0xAB` aus, und der Debugger (hier QEMU) erkennt ihn als Bitte, einen Text auszugeben oder das
Programm zu beenden. Das braucht keine Treiber und keine serielle Schnittstelle.

## Auf einen echten Chip übertragen

1. **Ziel wählen.** Cortex-M4F und M7 mit Fließkommaeinheit: `thumbv7em-none-eabihf`. Cortex-M3 (und
   M4 ohne FPU): `thumbv7m-none-eabi`, das läuft im Emulator, wie oben gezeigt. Cortex-M0/M0+:
   `thumbv6m-none-eabi`. Für diese beiden rechnet der Compiler Fließkomma in Software, das ist
   langsamer, ändert aber keine Zahl.
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
