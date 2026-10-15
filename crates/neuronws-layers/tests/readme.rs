//! Die README darf der Wirklichkeit nicht davonlaufen: Das Einstiegsbeispiel ist genau das Beispiel
//! `quickstart`, und alle Verweise auf Dateien im Projekt führen zu etwas, das es gibt.

use std::path::Path;

const README: &str = include_str!("../../../README.md");
const QUICKSTART: &str = include_str!("../examples/quickstart.rs");

#[test]
fn the_readme_shows_the_quickstart_example_verbatim() {
    assert!(
        README.contains(QUICKSTART.trim_end()),
        "Der Code im Abschnitt \"Einstieg\" der README muss genau dem Beispiel \
         crates/neuronws-layers/examples/quickstart.rs entsprechen. Wurde das Beispiel geändert, \
         den Text in der README nachziehen."
    );
}

/// Alle Ziele von Markdown-Verweisen der Form `](ziel)`, ohne Web-Adressen und ohne `#anker`.
fn local_links(markdown: &str) -> Vec<&str> {
    markdown
        .split("](")
        .skip(1)
        .filter_map(|rest| rest.split(')').next())
        .filter(|target| !target.starts_with("http") && !target.starts_with('#'))
        .collect()
}

#[test]
fn every_file_the_readme_links_to_exists() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let links = local_links(README);
    assert!(
        links.len() > 10,
        "es sollten viele Verweise zu finden sein: {links:?}"
    );
    for link in links {
        assert!(
            root.join(link).exists(),
            "Die README verweist auf {link}, das es nicht gibt"
        );
    }
}

#[test]
fn the_link_finder_skips_web_addresses_and_anchors() {
    let text = "[a](https://example.org) [b](#oben) [c](src/lib.rs) [d](mcu)";
    assert_eq!(local_links(text), ["src/lib.rs", "mcu"]);
}
