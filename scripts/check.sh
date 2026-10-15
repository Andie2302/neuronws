#!/usr/bin/env bash
# Prüft alles, was auch die CI prüft: Format, Clippy und Tests in allen Feature-Kombinationen.
#
#   scripts/check.sh          alles
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
done

echo "== Perceptron und Ziffern: Clippy und Tests"
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --quiet

if [ "$mode" != "quick" ]; then
    echo "== Dokumentation ohne Warnungen"
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --quiet
fi
echo "Alles in Ordnung."
