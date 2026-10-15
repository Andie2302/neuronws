#!/usr/bin/env bash
# Baut alle Beispiele im Release-Build und führt jedes aus. Ein Beispiel, das abstürzt oder mit
# einem Fehler endet, bricht das Skript ab.
#
#   scripts/examples.sh
#
# Die Beispiele rechnen mit `exp`, `tanh` und Co., deshalb wird `libm` eingeschaltet. Im Debug-Build
# wären manche (zum Beispiel `sequence_memory`) viel zu langsam.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Beispiele bauen (Release)"
cargo build --workspace --release --examples --features neuronws-layers/libm --quiet

echo "== Beispiele ausführen"
for source in crates/*/examples/*.rs; do
    name="$(basename "$source" .rs)"
    start=$SECONDS
    # Die Ausgabe interessiert hier nicht, nur ob das Beispiel durchläuft.
    if ! "target/release/examples/$name" >/dev/null 2>&1; then
        echo "Das Beispiel $name ist fehlgeschlagen. Ausgabe:" >&2
        "target/release/examples/$name" >&2 || true
        exit 1
    fi
    printf '  %-24s ok (%ss)\n' "$name" "$((SECONDS - start))"
done
