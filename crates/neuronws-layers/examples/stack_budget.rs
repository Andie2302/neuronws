//! Wie viel Stack braucht ein Netz? Misst es, statt zu raten.
//!
//! Jeder `Block` hält seine Eingabe, seine Ausgabe und den Gradienten für den Vorgänger als
//! Arrays, dazu Parameter und deren Gradienten. Ein Optimierer wie Adam legt zwei weitere Kopien
//! der Parameter an. Alles liegt auf dem Stack, solange man es nicht auf den Heap legt. Bei
//! Bildern wird das schnell viel: Eine Faltung mit 32 Kanälen auf 28×28 hat über 25 000 Werte pro
//! Array, und jedes Array gibt es mehrfach.
//!
//! Das Programm startet sich für jede Netzgröße selbst als Kind-Prozess mit einem Thread, dessen
//! Stack genau `n` KB groß ist, und sucht per Halbierung das kleinste `n`, bei dem ein
//! Lernschritt noch durchläuft. Ein Stacküberlauf lässt sich nicht abfangen, deshalb braucht es
//! den Umweg über einen eigenen Prozess. Die letzte Spalte zeigt, was `Box` bringt.
//!
//! Start (Debug und Release unterscheiden sich stark):
//!
//! ```text
//! cargo run -p neuronws-layers --features alloc,libm --example stack_budget
//! cargo run -p neuronws-layers --features alloc,libm --release --example stack_budget
//! ```

use std::mem::size_of_val;
use std::process::{Command, Stdio};

use neuronws_layers::activation::{Relu, Tanh};
use neuronws_layers::activation_layer::Elementwise;
use neuronws_layers::adam::AdamW;
use neuronws_layers::block::Block;
use neuronws_layers::chain::LayerExt;
use neuronws_layers::conv::Conv2d;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::init::{He, Xavier};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::neurons::neurons;
use neuronws_layers::pooling::{AvgPool2d, MaxPool2d};
use neuronws_layers::random::Rng;
use neuronws_layers::reshape::Reshape;
use neuronws_layers::softmax_layer::SoftmaxLayer;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::values::Image;

const CLASSES: usize = 10;

/// Ein Lernschritt über zwei Bilder, damit Vorwärts-, Rückwärtsrechnung und Optimierer laufen.
fn data() -> [(Image<1, 28, 28>, [f32; CLASSES]); 2] {
    let mut rng = Rng::new(1);
    core::array::from_fn(|i| {
        let image = core::array::from_fn(|_| {
            core::array::from_fn(|_| core::array::from_fn(|_| rng.uniform(0.0, 1.0)))
        });
        let mut target = [0.0; CLASSES];
        target[i] = 1.0;
        (image, target)
    })
}

/// Ein einfaches Netz aus zwei Dense-Schichten: 784 → 64 → 10.
///
/// Jede Konfiguration ist `#[inline(never)]`: Sonst fasst der Compiler im Release-Build alle in
/// einen Stack-Rahmen zusammen, und man mäße den Bedarf aller Netze zusammen statt eines einzelnen.
#[inline(never)]
fn mlp() -> usize {
    let mut rng = Rng::new(2);
    let mut net = Block::<Reshape<Image<1, 28, 28>, [f32; 784]>>::new()
        .then(DenseLayer::<784, 64>::xavier(&mut rng))
        .then(neuronws_layers::activation_layer::ActivationLayer::<64, _>::new(Tanh))
        .then(DenseLayer::<64, CLASSES>::xavier(&mut rng))
        .then(SoftmaxLayer::<CLASSES>::new());
    let mut trainer = Trainer::new(
        CategoricalCrossEntropy,
        AdamW::for_model(&net, 0.01, 0.0),
        2,
    );
    let size = size_of_val(&net) + size_of_val(trainer.optimizer());
    trainer.epoch(&mut net, &data());
    size
}

/// Zwei Faltungen mit je `C1` und `C2` Kanälen, jeweils mit ReLU und Pooling. Am Ende mittelt ein
/// Pooling über das ganze Bild, dann eine Dense-Schicht.
///
/// ```text
/// 1×28×28 → Conv 3×3 (C1) → ReLU → MaxPool → C1×14×14 → Conv 3×3 (C2) → ReLU → MaxPool
///         → C2×7×7 → Mittelwert über das Bild → C2 → Dense → Softmax 10
/// ```
#[inline(never)]
fn cnn<const C1: usize, const C2: usize>() -> usize {
    let mut rng = Rng::new(3);
    let mut net = neurons(
        Conv2d::<1, C1, 28, 28, 3, 3, 1, 1, 1, 28, 28>::new(),
        &He,
        Elementwise::new(Relu),
        &mut rng,
    )
    .then(Block::from_parts(
        MaxPool2d::<C1, 28, 28, 2, 14, 14>::new(),
        (),
    ))
    .then(neurons(
        Conv2d::<C1, C2, 14, 14, 3, 3, 1, 1, 1, 14, 14>::new(),
        &He,
        Elementwise::new(Relu),
        &mut rng,
    ))
    .then(Block::from_parts(
        MaxPool2d::<C2, 14, 14, 2, 7, 7>::new(),
        (),
    ))
    .then(Block::from_parts(AvgPool2d::<C2, 7, 7, 7, 1, 1>::new(), ()))
    .then(Block::from_parts(
        Reshape::<Image<C2, 1, 1>, [f32; C2]>::new(),
        (),
    ))
    .then(neurons(
        neuronws_layers::dense_layer::Linear::<C2, CLASSES>::new(),
        &Xavier,
        neuronws_layers::softmax_layer::Softmax::<CLASSES>::new(),
        &mut rng,
    ));
    let mut trainer = Trainer::new(
        CategoricalCrossEntropy,
        AdamW::for_model(&net, 0.01, 0.0),
        2,
    );
    let size = size_of_val(&net) + size_of_val(trainer.optimizer());
    trainer.epoch(&mut net, &data());
    size
}

/// Dasselbe Netz, aber jede Schicht einzeln auf dem Heap: Die Kette selbst besteht dann nur noch
/// aus Zeigern. Nur mit dem Feature `alloc`.
#[cfg(feature = "alloc")]
#[inline(never)]
fn cnn_boxed<const C1: usize, const C2: usize>() -> usize {
    let mut rng = Rng::new(3);
    let mut net = Box::new(Block::initialized(
        Conv2d::<1, C1, 28, 28, 3, 3, 1, 1, 1, 28, 28>::new(),
        &He,
        &mut rng,
    ))
    .then(Box::new(
        Block::<Elementwise<Image<C1, 28, 28>, Relu>>::new(Relu),
    ))
    .then(Box::new(Block::from_parts(
        MaxPool2d::<C1, 28, 28, 2, 14, 14>::new(),
        (),
    )))
    .then(Box::new(Block::initialized(
        Conv2d::<C1, C2, 14, 14, 3, 3, 1, 1, 1, 14, 14>::new(),
        &He,
        &mut rng,
    )))
    .then(Box::new(
        Block::<Elementwise<Image<C2, 14, 14>, Relu>>::new(Relu),
    ))
    .then(Box::new(Block::from_parts(
        MaxPool2d::<C2, 14, 14, 2, 7, 7>::new(),
        (),
    )))
    .then(Box::new(Block::from_parts(
        AvgPool2d::<C2, 7, 7, 7, 1, 1>::new(),
        (),
    )))
    .then(Box::new(Block::from_parts(
        Reshape::<Image<C2, 1, 1>, [f32; C2]>::new(),
        (),
    )))
    .then(Box::new(DenseLayer::<C2, CLASSES>::xavier(&mut rng)))
    .then(Box::new(SoftmaxLayer::<CLASSES>::new()));
    // Auch der Optimierer kann in einer Box liegen. Sein Gedächtnis ist so groß wie die Parameter.
    let optimizer = Box::new(AdamW::for_model(&net, 0.01, 0.0));
    let mut trainer = Trainer::new(CategoricalCrossEntropy, optimizer, 2);
    let size = size_of_val(&net) + size_of_val(trainer.optimizer());
    trainer.epoch(&mut net, &data());
    size
}

const CONFIGS: &[(&str, &str)] = &[
    ("mlp", "Dense 784→64→10"),
    ("cnn-8-16", "Faltung 8 und 16 Kanäle"),
    ("cnn-16-32", "Faltung 16 und 32 Kanäle"),
    ("cnn-32-64", "Faltung 32 und 64 Kanäle"),
    ("cnn-64-128", "Faltung 64 und 128 Kanäle"),
    #[cfg(feature = "alloc")]
    ("cnn-32-64-boxed", "wie 32/64, jede Schicht in einer Box"),
    #[cfg(feature = "alloc")]
    ("cnn-64-128-boxed", "wie 64/128, jede Schicht in einer Box"),
];

/// Läuft einen Lernschritt für die Konfiguration `name` und liefert die Größe von Netz und
/// Optimierer in Bytes (so wie sie im Speicher liegen: bei `Box` nur die Zeiger).
fn run(name: &str) -> usize {
    match name {
        "mlp" => mlp(),
        "cnn-8-16" => cnn::<8, 16>(),
        "cnn-16-32" => cnn::<16, 32>(),
        "cnn-32-64" => cnn::<32, 64>(),
        "cnn-64-128" => cnn::<64, 128>(),
        #[cfg(feature = "alloc")]
        "cnn-32-64-boxed" => cnn_boxed::<32, 64>(),
        #[cfg(feature = "alloc")]
        "cnn-64-128-boxed" => cnn_boxed::<64, 128>(),
        other => panic!("unbekannte Konfiguration {other}"),
    }
}

/// Startet dieses Programm als Kind, das `name` in einem Thread mit `kb` KB Stack ausführt.
fn works_with(name: &str, kb: usize) -> bool {
    Command::new(std::env::current_exe().expect("eigener Pfad"))
        .args(["child", name, &kb.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Das kleinste `kb`, bei dem `name` noch läuft (Halbierung, `None` bis 256 MB nicht).
fn minimum_stack_kb(name: &str) -> Option<usize> {
    let (mut low, mut high) = (16, 262_144);
    if !works_with(name, high) {
        return None;
    }
    while low < high {
        let middle = usize::midpoint(low, high);
        if works_with(name, middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Some(low)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Kind-Modus: genau eine Konfiguration in einem Thread mit festem Stack.
    if args.len() == 4 && args[1] == "child" {
        let name = args[2].clone();
        let kb: usize = args[3].parse().expect("Stack in KB");
        std::thread::Builder::new()
            .stack_size(kb * 1024)
            .spawn(move || {
                run(&name);
            })
            .expect("Thread")
            .join()
            .expect("Thread beendet");
        return;
    }

    let profile = if cfg!(debug_assertions) {
        "Debug"
    } else {
        "Release"
    };
    println!("Stack-Bedarf eines Lernschrittes ({profile}-Build, f32, AdamW)\n");
    println!(
        "{:18} {:28} {:>14} {:>16}",
        "Name", "Netz", "Größe im RAM", "Stack mindestens"
    );
    for (name, description) in CONFIGS {
        // Größe in einem eigenen großen Thread bestimmen, damit schon das Messen nicht überläuft.
        let name_owned = (*name).to_string();
        let size = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(move || run(&name_owned))
            .expect("Thread")
            .join()
            .expect("Thread beendet");
        let needed = minimum_stack_kb(name)
            .map_or_else(|| "über 256 MB".to_string(), |kb| format!("{kb} KB"));
        println!(
            "{name:18} {description:28} {:>11} KB {needed:>16}",
            size / 1024
        );
    }
    println!("\nZum Vergleich: Ein Test-Thread hat 2048 KB, der Haupt-Thread meist 8192 KB.");
}
