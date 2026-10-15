#!/usr/bin/env bash
# Baut das Testprogramm für den Mikrocontroller und führt es im Emulator QEMU aus.
#
#   scripts/mcu.sh
#
# Das Programm in `mcu/` ist ein echtes Bare-Metal-Programm für einen Cortex-M4F mit
# Fließkommaeinheit (`thumbv7em-none-eabihf`): ohne Betriebssystem, ohne `std`, ohne `alloc`. Es
# lernt XOR, speichert und lädt die Gewichte und prüft das Perceptron. QEMU ersetzt dabei die
# Hardware, siehe mcu/README.md. Dafür braucht es `qemu-system-arm`
# (Ubuntu/Debian: sudo apt-get install qemu-system-arm).
set -euo pipefail
cd "$(dirname "$0")/.."

target="thumbv7em-none-eabihf"
if ! command -v qemu-system-arm >/dev/null; then
    echo "qemu-system-arm fehlt (Ubuntu/Debian: sudo apt-get install qemu-system-arm)" >&2
    exit 2
fi
if ! rustup target list --installed | grep -qx "$target"; then
    echo "== Ziel $target installieren"
    rustup target add "$target"
fi

echo "== MCU: Format, Clippy, Build"
# `--target` steht auch in mcu/.cargo/config.toml, aber Cargo liest diese Datei nur, wenn man im
# Ordner `mcu/` arbeitet. Hier läuft alles vom Hauptordner aus, deshalb ausdrücklich.
cargo fmt --manifest-path mcu/Cargo.toml -- --check
cargo clippy --manifest-path mcu/Cargo.toml --target "$target" --release -- -D warnings
cargo build --manifest-path mcu/Cargo.toml --target "$target" --release --quiet

echo "== MCU: im Emulator ausführen (Cortex-M4F, Maschine mps2-an386)"
# QEMU schreibt die Semihosting-Ausgabe auf stderr, deshalb `2>&1`.
status=0
output=$(timeout 120 qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -nographic \
    -semihosting-config enable=on,target=native \
    -kernel "mcu/target/$target/release/neuronws-mcu" 2>&1) || status=$?
echo "$output"
if [ "$status" -ne 0 ]; then
    echo "Das Programm endete mit Exit-Code $status" >&2
    exit 1
fi
grep -qx "ALLES OK" <<<"$output"

# Dieselbe Rechnung auf dem Rechner (x86-64 o. ä.) muss dieselbe Zahl liefern. Dank `libm` rechnet
# jede Plattform mit demselben Code.
echo "== MCU: Vergleich mit dem Rechner"
host=$(cargo run --quiet -p neuronws-layers --features libm --example quickstart | head -n 1)
mcu=$(grep -x "Fehler vorher .*" <<<"$output")
if [ "$host" != "$mcu" ]; then
    echo "Rechner:         $host" >&2
    echo "Mikrocontroller: $mcu" >&2
    echo "Die Ergebnisse unterscheiden sich." >&2
    exit 1
fi
echo "Gleiches Ergebnis auf beiden: $host"
echo "MCU-Test in Ordnung."
