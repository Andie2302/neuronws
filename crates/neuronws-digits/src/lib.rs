//! Ziffernerkennung auf 5×7-Bitmaps: die Daten und ein passendes Netz.
//!
//! Jede Ziffer ist ein Bild mit 5 Spalten und 7 Zeilen, also 35 Pixel. Ein Pixel ist 1.0
//! (schwarz) oder 0.0 (weiß). Das Netz bekommt die 35 Werte und liefert 10 Wahrscheinlichkeiten,
//! eine pro Ziffer.

#![no_std]

use neuronws_layers::activation::Tanh;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::{Chain, LayerExt};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::random::Rng;
use neuronws_layers::softmax_layer::SoftmaxLayer;
use neuronws_layers::traits::ForwardPath;

pub const WIDTH: usize = 5;
pub const HEIGHT: usize = 7;
pub const PIXELS: usize = WIDTH * HEIGHT;
pub const CLASSES: usize = 10;

pub type Image = [f32; PIXELS];
pub type Label = [f32; CLASSES];
pub type Sample = (Image, Label);

/// Die zehn Ziffern als Text: `#` ist ein schwarzer, `.` ein weißer Pixel.
const GLYPHS: [[&str; HEIGHT]; CLASSES] = [
    [
        ".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###.",
    ], // 0
    [
        "..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###.",
    ], // 1
    [
        ".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####",
    ], // 2
    [
        ".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###.",
    ], // 3
    [
        "...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#.",
    ], // 4
    [
        "#####", "#....", "####.", "....#", "....#", "#...#", ".###.",
    ], // 5
    [
        ".###.", "#....", "#....", "####.", "#...#", "#...#", ".###.",
    ], // 6
    [
        "#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#...",
    ], // 7
    [
        ".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###.",
    ], // 8
    [
        ".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##..",
    ], // 9
];

const fn parse(rows: [&str; HEIGHT]) -> Image {
    let mut image = [0.0; PIXELS];
    let mut row = 0;
    while row < HEIGHT {
        let bytes = rows[row].as_bytes();
        let mut column = 0;
        while column < WIDTH {
            if bytes[column] == b'#' {
                image[row * WIDTH + column] = 1.0;
            }
            column += 1;
        }
        row += 1;
    }
    image
}

const fn parse_all() -> [Image; CLASSES] {
    let mut images = [[0.0; PIXELS]; CLASSES];
    let mut digit = 0;
    while digit < CLASSES {
        images[digit] = parse(GLYPHS[digit]);
        digit += 1;
    }
    images
}

/// Die zehn sauberen Bilder, `DIGITS[7]` ist die Sieben.
pub const DIGITS: [Image; CLASSES] = parse_all();

/// Das gewünschte Ergebnis für eine Ziffer: eine 1 an ihrer Stelle, sonst 0.
pub fn one_hot(digit: usize) -> Label {
    core::array::from_fn(|i| f32::from(i == digit))
}

/// Die Stelle mit dem größten Wert, also die erkannte Ziffer.
pub fn argmax(output: &Label) -> usize {
    (0..CLASSES)
        .max_by(|&a, &b| output[a].total_cmp(&output[b]))
        .unwrap_or(0)
}

/// Kippt genau `count` verschiedene, zufällig gewählte Pixel (schwarz wird weiß und umgekehrt).
pub fn flip_pixels(image: &Image, count: usize, rng: &mut Rng) -> Image {
    assert!(count <= PIXELS, "es gibt nur {PIXELS} Pixel");
    let mut flipped = *image;
    let mut already_flipped = [false; PIXELS];
    let mut done = 0;
    while done < count {
        let pixel = rng.next_below(PIXELS);
        if !already_flipped[pixel] {
            already_flipped[pixel] = true;
            flipped[pixel] = 1.0 - flipped[pixel];
            done += 1;
        }
    }
    flipped
}

/// `N` Beispiele, reihum für die Ziffern 0 bis 9, jeweils mit 0 bis `max_flips` gekippten Pixeln.
///
/// # Panics
/// Wenn `N` nicht durch 10 teilbar ist.
pub fn noisy_dataset<const N: usize>(max_flips: usize, rng: &mut Rng) -> [Sample; N] {
    assert!(
        N.is_multiple_of(CLASSES),
        "N muss durch {CLASSES} teilbar sein"
    );
    core::array::from_fn(|i| {
        let digit = i % CLASSES;
        let flips = rng.next_below(max_flips + 1);
        (flip_pixels(&DIGITS[digit], flips, rng), one_hot(digit))
    })
}

/// Anteil der Beispiele, bei denen die größte Ausgabe an der richtigen Stelle steht.
pub fn accuracy<M>(model: &M, data: &[Sample]) -> f32
where
    M: ForwardPath<ForwardInput = Image, ForwardOutput = Label>,
{
    let correct = data
        .iter()
        .filter(|(image, label)| argmax(&model.forward(image)) == argmax(label))
        .count();
    correct as f32 / data.len() as f32
}

/// 35 Pixel → 16 verborgene Neuronen (Tanh) → 10 Ziffern (Softmax).
pub type Network = Chain<
    Chain<
        Chain<DenseLayer<PIXELS, HIDDEN>, ActivationLayer<HIDDEN, Tanh>>,
        DenseLayer<HIDDEN, CLASSES>,
    >,
    SoftmaxLayer<CLASSES>,
>;

pub const HIDDEN: usize = 16;

/// Ein neues Netz mit zufälligen Startgewichten (Xavier).
pub fn new_network(rng: &mut Rng) -> Network {
    DenseLayer::<PIXELS, HIDDEN>::xavier(rng)
        .then(ActivationLayer::<HIDDEN, _>::new(Tanh))
        .then(DenseLayer::<HIDDEN, CLASSES>::xavier(rng))
        .then(SoftmaxLayer::<CLASSES>::new())
}
