#!/usr/bin/env bash
# Baut das Testprogramm für Mikrocontroller und führt es im Emulator QEMU aus.
#
#   scripts/mcu.sh
#
# Das Programm in `mcu/` ist ein echtes Bare-Metal-Programm: ohne Betriebssystem, ohne `std`, ohne
# `alloc`. Es lernt XOR, speichert und lädt die Gewichte und prüft das Perceptron. Es läuft auf zwei
# Prozessoren:
#
#   thumbv7em-none-eabihf   Cortex-M4F, Fließkomma in der Hardware    (QEMU-Maschine mps2-an386)
#   thumbv7m-none-eabi      Cortex-M3, Fließkomma in Software          (QEMU-Maschine mps2-an385)
#
# QEMU ersetzt dabei die Hardware, siehe mcu/README.md. Dafür braucht es `qemu-system-arm`
# (Ubuntu/Debian: sudo apt-get install qemu-system-arm).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v qemu-system-arm >/dev/null; then
    echo "qemu-system-arm fehlt (Ubuntu/Debian: sudo apt-get install qemu-system-arm)" >&2
    exit 2
fi

# Ziel, QEMU-Maschine und Prozessor
runs=(
    "thumbv7em-none-eabihf mps2-an386 cortex-m4"
    "thumbv7m-none-eabi mps2-an385 cortex-m3"
)

for run in "${runs[@]}"; do
    read -r target machine cpu <<<"$run"
    if ! rustup target list --installed | grep -qx "$target"; then
        echo "== Ziel $target installieren"
        rustup target add "$target"
    fi
done

# Dieselbe Rechnung auf dem Rechner (x86-64 o. ä.) muss auf jedem Prozessor dieselbe Zahl liefern.
# Dank `libm` rechnet jede Plattform mit demselben Code.
host=$(cargo run --quiet -p neuronws-layers --features libm --example quickstart | head -n 1)
# Und die beiden Mikrocontroller müssen sich bis auf das letzte Bit der Ausgaben einig sein, obwohl
# der eine in der Hardware und der andere in Software rechnet.
first_bits=""

echo "== MCU: Format"
cargo fmt --manifest-path mcu/Cargo.toml -- --check

for run in "${runs[@]}"; do
    read -r target machine cpu <<<"$run"

    echo "== MCU $target: Clippy und Build"
    # `--target` steht auch in mcu/.cargo/config.toml, aber Cargo liest diese Datei nur, wenn man im
    # Ordner `mcu/` arbeitet. Hier läuft alles vom Hauptordner aus, deshalb ausdrücklich.
    cargo clippy --manifest-path mcu/Cargo.toml --target "$target" --release -- -D warnings
    cargo build --manifest-path mcu/Cargo.toml --target "$target" --release --quiet

    echo "== MCU $target: im Emulator ausführen ($cpu, Maschine $machine)"
    # QEMU schreibt die Semihosting-Ausgabe auf stderr, deshalb `2>&1`.
    status=0
    output=$(timeout 120 qemu-system-arm -machine "$machine" -cpu "$cpu" -nographic \
        -semihosting-config enable=on,target=native \
        -kernel "mcu/target/$target/release/neuronws-mcu" 2>&1) || status=$?
    echo "$output"
    if [ "$status" -ne 0 ]; then
        echo "Das Programm endete mit Exit-Code $status" >&2
        exit 1
    fi
    grep -qx "ALLES OK" <<<"$output"

    mcu=$(grep -x "Fehler vorher .*" <<<"$output")
    if [ "$host" != "$mcu" ]; then
        echo "Rechner:         $host" >&2
        echo "Mikrocontroller: $mcu" >&2
        echo "Die Ergebnisse unterscheiden sich." >&2
        exit 1
    fi
    echo "Gleiches Ergebnis wie auf dem Rechner: $host"

    bits=$(grep "Bits" <<<"$output")
    if [ -z "$first_bits" ]; then
        first_bits="$bits"
    elif [ "$first_bits" != "$bits" ]; then
        echo "Die Ausgaben unterscheiden sich im letzten Bit zwischen den Prozessoren:" >&2
        diff <(echo "$first_bits") <(echo "$bits") >&2 || true
        exit 1
    else
        echo "Alle vier Ausgaben sind bitgleich zu denen des ersten Prozessors."
    fi
done
echo "MCU-Test in Ordnung."
