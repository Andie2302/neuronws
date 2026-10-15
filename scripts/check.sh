#!/usr/bin/env bash
# Prüft alles, was auch die CI prüft: Format, Clippy und Tests in allen Feature-Kombinationen.
#
#   scripts/check.sh          alles (auch Beispiele und der Test im Mikrocontroller-Emulator)
#   scripts/check.sh quick    nur "keine Features" und "libm" (für zwischendurch)
#
# Die Bibliothek ist standardmäßig `no_std` ohne `alloc` und ohne `std`. Deshalb wird jede
# Kombination ohne Standard-Features gebaut und geprüft.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-full}"
if [ "$mode" = "quick" ]; then
    combos=("" "libm")
else
    combos=("" "std" "libm" "std,libm" "alloc" "alloc,std,libm")
fi

echo "== Format"
cargo fmt --all -- --check

for features in "${combos[@]}"; do
    flags=(--no-default-features)
    [ -n "$features" ] && flags+=(--features "$features")
    label="${features:-keine Features}"

    echo "== neuronws-layers: Clippy ($label)"
    cargo clippy -p neuronws-layers "${flags[@]}" --all-targets -- -D warnings
    echo "== neuronws-layers: Tests ($label)"
    cargo test -p neuronws-layers "${flags[@]}" --quiet
    if [ "$mode" != "quick" ]; then
        # Verweise in der Dokumentation auf Typen, die es ohne ein Feature nicht gibt, sind ein
        # Fehler. Deshalb wird die Dokumentation für jede Kombination einzeln gebaut.
        echo "== neuronws-layers: Dokumentation ($label)"
        RUSTDOCFLAGS="-D warnings" cargo doc -p neuronws-layers "${flags[@]}" --no-deps --quiet
    fi
done

echo "== Perceptron und Ziffern: Clippy und Tests"
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --quiet

if [ "$mode" != "quick" ]; then
    echo "== Dokumentation ohne Warnungen (alle Crates)"
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --quiet
    RUSTDOCFLAGS="-D warnings" cargo doc -p neuronws-digits --no-default-features --no-deps --quiet

    # Der eigentliche Beweis für `no_std` ohne `alloc`: ein Ziel ohne Betriebssystem. Dort gibt es
    # weder `std` noch einen Heap, jede versehentliche Benutzung wäre ein Fehler.
    target="thumbv7em-none-eabihf"
    if ! rustup target list --installed | grep -qx "$target"; then
        echo "== Ziel $target installieren"
        rustup target add "$target"
    fi
    for features in "" "libm" "alloc" "alloc,libm"; do
        flags=(--no-default-features)
        [ -n "$features" ] && flags+=(--features "$features")
        echo "== Bare-Metal-Build neuronws-layers (${features:-keine Features})"
        cargo build -p neuronws-layers "${flags[@]}" --target "$target" --quiet
    done
    echo "== Bare-Metal-Build neuronws-perceptrons und neuronws-digits"
    cargo build -p neuronws-perceptrons --target "$target" --quiet
    cargo build -p neuronws-digits --no-default-features --target "$target" --quiet
    cargo build -p neuronws-digits --target "$target" --quiet

    scripts/examples.sh

    # Läuft das Programm wirklich ohne Betriebssystem? Dafür braucht es den Emulator QEMU.
    if command -v qemu-system-arm >/dev/null; then
        scripts/mcu.sh
    else
        echo "== MCU-Test übersprungen: qemu-system-arm fehlt (siehe scripts/mcu.sh)"
    fi
fi
echo "Alles in Ordnung."
