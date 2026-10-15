//! Ein Perceptron lernt Logikgatter aus Beispielen, scheitert an XOR, und zwei Schichten lösen es.
//!
//! Das Beispiel hat drei Teile:
//!
//! 1. UND, ODER und NAND sind **linear trennbar**: Eine Gerade schneidet die Einsen von den
//!    Nullen. Das Perceptron findet sie, und der Text-Plot zeigt, wo sie liegt.
//! 2. XOR ist es nicht. Das Perceptron macht in jeder Runde Fehler, ohne dass es je besser wird.
//! 3. Zwei Schichten lösen XOR: `OR` und `NAND` sind einzeln lernbar, und ein drittes Perceptron
//!    kombiniert ihre Antworten mit `AND`.
//!
//! Start: `cargo run -p neuronws-perceptrons --example learn_gates`

use neuronws_layers::datasets::logic::{AND, NAND, OR, Table, XOR};
use neuronws_perceptrons::Perceptron;

/// Die Gerade `w₁·x + w₂·y + b = 0` als Gleichung.
fn describe(perceptron: &Perceptron<2>) -> String {
    let [w1, w2] = *perceptron.weights();
    format!("{w1:.2}·x {w2:+.2}·y {:+.2} ≥ 0  →  1", perceptron.bias())
}

/// Zeichnet die Ebene von x = −0,5 bis 1,5 und y = −0,5 bis 1,5 (die Zeilen im Abstand 0,25, damit
/// die Beispiele bei y = 0 und y = 1 auf eine Zeile fallen). Die Seite, auf der das
/// Perceptron "1" sagt, ist mit Punkten gefüllt. `X` ist ein Beispiel mit Ziel 1, `o` eines mit
/// Ziel 0. Richtig gelernt, liegen alle `X` im gefüllten Bereich und alle `o` außerhalb.
fn plot(perceptron: &Perceptron<2>, data: &Table) {
    const COLUMNS: usize = 21;
    const ROWS: usize = 9;
    for row in 0..ROWS {
        let y = 1.5 - row as f32 * 0.25;
        let line: String = (0..COLUMNS)
            .map(|column| {
                let x = -0.5 + column as f32 * 0.1;
                let point = data
                    .iter()
                    .find(|([px, py], _)| (px - x).abs() < 0.05 && (py - y).abs() < 0.125);
                match point {
                    Some((_, target)) if *target == 1.0 => 'X',
                    Some(_) => 'o',
                    None if perceptron.predict(&[x, y]) == 1.0 => '·',
                    None => ' ',
                }
            })
            .collect();
        println!("      y={y:4.1} |{line}|");
    }
}

fn learn_separable(name: &str, data: &Table) {
    let mut perceptron = Perceptron::<2>::new([0.0, 0.0], 0.0);
    match perceptron.fit(data, 1000, 0.1) {
        Some(epochs) => println!(
            "{name}: gelernt nach {epochs} Runden, {}",
            describe(&perceptron)
        ),
        None => println!("{name}: nicht gelernt"),
    }
    plot(&perceptron, data);
    println!();
}

fn main() {
    println!("=== 1. Linear trennbar: eine Gerade genügt ===\n");
    learn_separable("UND ", &AND);
    learn_separable("ODER", &OR);
    learn_separable("NAND", &NAND);

    println!("=== 2. XOR: keine Gerade trennt die Einsen von den Nullen ===\n");
    let mut single = Perceptron::<2>::new([0.0, 0.0], 0.0);
    println!("Fehler (von 4 Beispielen) nach jeder Runde:");
    let mut history = Vec::new();
    for _ in 0..16 {
        for (input, target) in &XOR {
            single.learn(input, *target, 0.1);
        }
        history.push(single.mistakes(&XOR));
    }
    println!("  {history:?}");
    println!(
        "Die Zahl der Fehler wird nie 0. Eine Gerade kann höchstens drei von vier Punkten richtig\n\
         einordnen, und die Lernregel dreht sie bei jedem Fehler weiter, ohne je zur Ruhe zu kommen.\n"
    );

    println!("=== 3. Zwei Schichten lösen XOR: XOR = ODER und NAND ===\n");
    let mut or = Perceptron::<2>::new([0.0, 0.0], 0.0);
    let mut nand = Perceptron::<2>::new([0.0, 0.0], 0.0);
    or.fit(&OR, 1000, 0.1).expect("OR ist linear trennbar");
    nand.fit(&NAND, 1000, 0.1)
        .expect("NAND ist linear trennbar");

    // Die zweite Stufe sieht nicht x und y, sondern die Antworten der ersten beiden Perceptrons.
    let hidden = |input: &[f32; 2]| [or.predict(input), nand.predict(input)];
    let second_stage: Vec<([f32; 2], f32)> = XOR
        .iter()
        .map(|(input, target)| (hidden(input), *target))
        .collect();
    let mut and = Perceptron::<2>::new([0.0, 0.0], 0.0);
    let rounds = and
        .fit(&second_stage, 1000, 0.1)
        .expect("AND auf den Antworten ist linear trennbar");
    println!(
        "Die dritte Stufe lernt UND nach {rounds} Runden: {}\n",
        describe(&and)
    );

    println!("  x  y | ODER NAND | UND(ODER, NAND) | Ziel");
    for (input, target) in &XOR {
        let [h_or, h_nand] = hidden(input);
        println!(
            "  {}  {} |   {}    {}  |        {}        |  {}",
            input[0],
            input[1],
            h_or,
            h_nand,
            and.predict(&[h_or, h_nand]),
            target
        );
    }
    println!(
        "\nDie ersten beiden Perceptrons verbiegen die Ebene so, dass die dritte Stufe die Punkte\n\
         wieder mit einer Gerade trennen kann. Das ist die Idee hinter jedem mehrschichtigen Netz."
    );
}
