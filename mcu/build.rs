//! Sagt dem Linker, wo das Linker-Skript liegt.

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("Cargo setzt CARGO_MANIFEST_DIR");
    println!("cargo:rustc-link-arg=-T{dir}/link.ld");
    println!("cargo:rerun-if-changed=link.ld");
    println!("cargo:rerun-if-changed=build.rs");
}
