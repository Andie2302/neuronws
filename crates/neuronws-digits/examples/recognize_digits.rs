//! Ein Netz lernt, Ziffern auf 5×7-Bildern zu erkennen, auch wenn einzelne Pixel falsch sind.
//!
//! Start: `cargo run --release -p neuronws-digits --example recognize_digits`

use neuronws_digits::{
    CLASSES, DIGITS, HEIGHT, Image, WIDTH, accuracy, argmax, flip_pixels, new_network,
    noisy_dataset, one_hot,
};
use neuronws_layers::loss::CategoricalCrossEntropy;
use neuronws_layers::optimizer::Sgd;
use neuronws_layers::random::Rng;
use neuronws_layers::training::train_epoch;
use neuronws_layers::traits::ForwardPath;

fn print_image(image: &Image) {
    for row in 0..HEIGHT {
        let line: String = (0..WIDTH)
            .map(|column| {
                if image[row * WIDTH + column] > 0.5 {
                    '#'
                } else {
                    '.'
                }
            })
            .collect();
        println!("    {line}");
    }
}

fn main() {
    let mut rng = Rng::new(2026);

    // Lernen: 300 Bilder, jedes mit 0 bis 2 falschen Pixeln.
    let mut training = noisy_dataset::<300>(2, &mut rng);
    // Prüfen: andere, frisch verrauschte Bilder, die das Netz nie gesehen hat.
    let test = noisy_dataset::<200>(2, &mut rng);

    let mut net = new_network(&mut rng);
    let mut sgd = Sgd::new(0.1);

    println!(
        "Vor dem Lernen: {:.0} % richtig",
        100.0 * accuracy(&net, &test)
    );
    for epoch in 1..=60 {
        rng.shuffle(&mut training);
        let loss = train_epoch(&mut net, &mut sgd, &CategoricalCrossEntropy, &training, 10);
        if epoch % 10 == 0 {
            println!(
                "Epoche {epoch:2}: Verlust {loss:.4}, Test {:.1} % richtig",
                100.0 * accuracy(&net, &test)
            );
        }
    }

    println!("\nSaubere Ziffern:");
    let clean: Vec<_> = DIGITS
        .iter()
        .enumerate()
        .map(|(d, image)| (*image, one_hot(d)))
        .collect();
    println!("  {:.0} % richtig", 100.0 * accuracy(&net, &clean));

    println!("\nWie viele falsche Pixel verträgt das Netz? (je 200 Bilder)");
    for flips in 0..=6 {
        let noisy: Vec<_> = (0..200)
            .map(|i| {
                (
                    flip_pixels(&DIGITS[i % CLASSES], flips, &mut rng),
                    one_hot(i % CLASSES),
                )
            })
            .collect();
        println!(
            "  {flips} falsche Pixel: {:.1} % richtig",
            100.0 * accuracy(&net, &noisy)
        );
    }

    println!("\nEinige Beispiele aus dem Test:");
    for (image, label) in test.iter().take(6) {
        let output = net.forward(image);
        let guess = argmax(&output);
        print_image(image);
        println!(
            "    -> erkannt: {guess} mit {:.0} % (richtig wäre {})\n",
            100.0 * output[guess],
            argmax(label)
        );
    }
}
