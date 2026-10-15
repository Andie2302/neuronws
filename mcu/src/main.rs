//! Läuft die Bibliothek wirklich ohne Betriebssystem? Dieses Programm tut es: Es lernt XOR, speichert
//! und lädt die Gewichte, prüft ein Perceptron und misst, wie viel Stack und Flash dafür nötig sind.
//! Alles auf einem Cortex-M4F mit Fließkommaeinheit (`thumbv7em-none-eabihf`), ohne `std` und
//! ohne `alloc`.
//!
//! Gestartet wird es im Emulator QEMU (siehe `README.md`). Die Ausgabe geht über "Semihosting":
//! Das Programm ruft den Debugger, hier QEMU, mit einem `bkpt`-Befehl, und der schreibt den Text
//! auf den Bildschirm des Rechners. So braucht es weder Treiber noch eine serielle Schnittstelle.
//! Bei Erfolg endet es mit dem Exit-Code 0, sonst mit 1.
//!
//! Was hier nicht aus dem Projekt kommt, ist der Startcode, den sonst die Bibliothek `cortex-m-rt`
//! liefert (Vektortabelle, Reset, FPU einschalten). Er steht in diesem Programm, damit das ganze
//! Projekt ohne fremde Bibliotheken auskommt, bis auf `libm`.

#![no_std]
#![no_main]

use core::arch::asm;
use core::fmt::{self, Write};
use core::mem::{size_of, size_of_val};
use core::panic::PanicInfo;

use neuronws_layers::activation::{Sigmoid, Tanh};
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::adam::Adam;
use neuronws_layers::chain::{Chain, LayerExt};
use neuronws_layers::datasets::logic;
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::loss::BinaryCrossEntropy;
use neuronws_layers::persist::{LoadError, byte_len, load_parameters, save_parameters};
use neuronws_layers::random::Rng;
use neuronws_layers::trainer::Trainer;
use neuronws_layers::traits::{ForwardPath, Trainable};
use neuronws_perceptrons::Perceptron;

// ---------------------------------------------------------------------------------------------
// Ausgabe und Ende über Semihosting
// ---------------------------------------------------------------------------------------------

const SYS_WRITE0: u32 = 0x04;
const SYS_EXIT_EXTENDED: u32 = 0x20;
const ADP_STOPPED_APPLICATION_EXIT: u32 = 0x20026;

/// Ruft den Debugger: `bkpt 0xAB` mit der Nummer der Operation in `r0` und einem Zeiger auf ihre
/// Parameter in `r1`. Das Ergebnis kommt in `r0` zurück.
fn semihosting(operation: u32, parameter: usize) -> u32 {
    let result;
    // SAFETY: `bkpt 0xAB` ist der Semihosting-Aufruf der ARM-Architektur. Er liest nur den
    // Speicher, auf den `parameter` zeigt. Ohne Debugger oder Emulator bleibt das Programm hier
    // hängen, das gilt für dieses Testprogramm.
    unsafe {
        asm!(
            "bkpt 0xAB",
            inlateout("r0") operation => result,
            in("r1") parameter,
            options(nostack),
        );
    }
    result
}

/// Eine Zeile Text: `core::fmt` schreibt hinein, `flush` schickt sie an den Host.
struct Line {
    bytes: [u8; 160],
    len: usize,
}

impl Line {
    const fn new() -> Self {
        Self {
            bytes: [0; 160],
            len: 0,
        }
    }

    /// `SYS_WRITE0` erwartet Text, der mit einem Nullbyte endet.
    fn flush(&mut self) {
        self.bytes[self.len] = 0;
        semihosting(SYS_WRITE0, self.bytes.as_ptr() as usize);
        self.len = 0;
    }
}

impl Write for Line {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for &byte in text.as_bytes() {
            if self.len == self.bytes.len() - 1 {
                self.flush();
            }
            self.bytes[self.len] = byte;
            self.len += 1;
        }
        Ok(())
    }
}

macro_rules! println {
    ($($argument:tt)*) => {{
        let mut line = Line::new();
        let _ = writeln!(line, $($argument)*);
        line.flush();
    }};
}

/// Beendet den Emulator mit dem Exit-Code 0 (Erfolg) oder 1.
fn exit(success: bool) -> ! {
    let block = [ADP_STOPPED_APPLICATION_EXIT, u32::from(!success)];
    semihosting(SYS_EXIT_EXTENDED, block.as_ptr() as usize);
    // Der Emulator beendet sich bei diesem Aufruf. Ein Debugger, der es nicht tut, lässt das
    // Programm hier warten.
    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("PANIK: {info}");
    exit(false)
}

// ---------------------------------------------------------------------------------------------
// Start: Vektortabelle, Reset, Fließkommaeinheit
// ---------------------------------------------------------------------------------------------

/// Der Einsprung nach dem Einschalten. Das Linker-Skript legt seine Adresse als zweites Wort in
/// die Vektortabelle (das erste ist der Stack-Zeiger).
#[unsafe(link_section = ".vector_table.reset_vector")]
#[used]
static RESET_VECTOR: extern "C" fn() -> ! = reset;

/// Alle anderen Ausnahmen (NMI, HardFault, ...) führen zur gleichen Meldung.
#[unsafe(link_section = ".vector_table.exceptions")]
#[used]
static EXCEPTIONS: [extern "C" fn() -> !; 14] = [fault; 14];

extern "C" fn fault() -> ! {
    println!("FEHLER: Der Prozessor hat eine Ausnahme gemeldet (zum Beispiel HardFault)");
    exit(false)
}

#[unsafe(no_mangle)]
pub extern "C" fn reset() -> ! {
    // Ohne diesen Schritt löst jeder Fließkomma-Befehl eine Ausnahme aus: Das Register CPACR
    // (Coprocessor Access Control, Adresse 0xE000ED88) gibt die Fließkommaeinheit (Coprozessoren
    // 10 und 11) erst frei, wenn man es beschreibt.
    // SAFETY: Das ist ein Register des Prozessorkerns, das auf jedem Cortex-M4F an dieser Adresse
    // liegt. Dies ist der erste Code nach dem Einschalten, es läuft nichts anderes.
    unsafe {
        core::ptr::write_volatile(0xE000_ED88 as *mut u32, 0b1111 << 20);
        asm!("dsb", "isb", options(nostack, preserves_flags));
    }
    exit(run())
}

// ---------------------------------------------------------------------------------------------
// Stack und Flash messen
// ---------------------------------------------------------------------------------------------

unsafe extern "C" {
    // Vom Linker-Skript: Anfang des Arbeitsspeichers, Ende des Stacks und Ende des Flash-Inhalts.
    static _ram_start: u8;
    static _stack_top: u8;
    static _flash_end: u8;
}

const PAINT: u32 = 0xDEAD_BEEF;

/// Bemalt den freien Teil des Stacks mit einem Muster. Danach zeigt `stack_used`, wie weit das
/// Programm ihn überschrieben hat.
#[inline(never)]
fn paint_stack() {
    let stack_pointer: usize;
    // SAFETY: Liest nur das Register `sp`.
    unsafe { asm!("mov {}, sp", out(reg) stack_pointer, options(nomem, nostack, preserves_flags)) };
    let start = &raw const _ram_start as usize;
    // Alles unterhalb des aktuellen Stack-Zeigers (mit Abstand für die Aufrufe der Schleife
    // selbst) ist frei und darf bemalt werden.
    let end = (stack_pointer - 256) & !3;
    for address in (start..end).step_by(4) {
        // SAFETY: `address` liegt im Arbeitsspeicher, ist durch 4 teilbar und unterhalb des
        // belegten Stacks.
        unsafe { core::ptr::write_volatile(address as *mut u32, PAINT) };
    }
}

/// Wie viele Bytes Stack seit `paint_stack` benutzt wurden (vom Ende des Arbeitsspeichers an
/// gerechnet, also einschließlich der Aufrufer von `paint_stack`).
fn stack_used() -> usize {
    let (start, top) = (
        &raw const _ram_start as usize,
        &raw const _stack_top as usize,
    );
    for address in (start..top).step_by(4) {
        // SAFETY: `address` liegt im Arbeitsspeicher und ist durch 4 teilbar.
        if unsafe { core::ptr::read_volatile(address as *const u32) } != PAINT {
            return top - address;
        }
    }
    top - start
}

// ---------------------------------------------------------------------------------------------
// Der eigentliche Test
// ---------------------------------------------------------------------------------------------

/// 2 Eingänge → 4 verborgene Neuronen (Tanh) → 1 Ausgang (Sigmoid), wie im Beispiel `quickstart`.
type Net = Chain<
    Chain<Chain<DenseLayer<2, 4>, ActivationLayer<4, Tanh>>, DenseLayer<4, 1>>,
    ActivationLayer<1, Sigmoid>,
>;

fn new_net(rng: &mut Rng) -> Net {
    DenseLayer::<2, 4>::xavier(rng)
        .then(ActivationLayer::<4, _>::new(Tanh))
        .then(DenseLayer::<4, 1>::xavier(rng))
        .then(ActivationLayer::<1, _>::new(Sigmoid))
}

/// Sammelt die Ergebnisse der Prüfungen.
struct Report {
    failed: usize,
}

impl Report {
    fn check(&mut self, name: &str, ok: bool) {
        println!("  [{}] {name}", if ok { "ok" } else { "FEHLER" });
        self.failed += usize::from(!ok);
    }
}

/// Ein Perceptron, das schon zur Compile-Zeit fertig ist: Die Gerade x + y = 1,5 ist UND. Es liegt
/// im Flash, nicht im Arbeitsspeicher.
const AND: Perceptron<2> = Perceptron::new([1.0, 1.0], -1.5);

/// Liefert `true`, wenn alles stimmt.
fn run() -> bool {
    paint_stack();
    let mut report = Report { failed: 0 };

    println!("neuronws auf einem Cortex-M4F: ohne Betriebssystem, ohne std, ohne alloc");

    // 1. Lernen: dasselbe wie im Beispiel `quickstart` (Fehler 0.831 -> 0.001), hier ohne
    //    Betriebssystem.
    let mut rng = Rng::new(1);
    let mut net = new_net(&mut rng);
    let data = logic::as_vectors(&logic::XOR);
    let mut trainer = Trainer::new(BinaryCrossEntropy, Adam::for_model(&net, 0.1), 4);
    println!(
        "Größen: Netz {} Bytes mit {} Parametern, Optimierer {} Bytes",
        size_of::<Net>(),
        <Net as Trainable>::PARAMETER_COUNT,
        size_of_val(&trainer),
    );

    let before: f32 = trainer.evaluate(&net, &data);
    let after: f32 = trainer.fit(&mut net, &data, 300);
    println!("Fehler vorher {before:.3}, nachher {after:.3}");
    report.check("Das Netz hat XOR gelernt (Fehler unter 0.05)", after < 0.05);
    for (input, target) in &data {
        let output = net.forward(input)[0];
        println!(
            "     {input:?} -> {output:.3} (soll {}, Bits {:#010x})",
            target[0],
            output.to_bits()
        );
        report.check(
            "Die Ausgabe liegt auf der richtigen Seite von 0.5",
            (output > 0.5) == (target[0] > 0.5),
        );
    }

    // 2. Gewichte speichern und laden, ohne Heap: Der Puffer liegt auf dem Stack.
    let mut buffer = [0_u8; byte_len::<Net>()];
    let written = save_parameters(&net, &mut buffer).expect("der Puffer hat die richtige Größe");
    println!("Gewichte gespeichert: {written} Bytes");
    let mut copy = new_net(&mut Rng::new(99));
    load_parameters(&mut copy, &buffer).expect("die Daten gehören zu diesem Netz");
    report.check(
        "Das geladene Netz rechnet bitgenau wie das gespeicherte",
        data.iter().all(|(x, _)| net.forward(x) == copy.forward(x)),
    );
    report.check(
        "Zu kurze Daten werden abgelehnt",
        load_parameters(&mut copy, &buffer[..10]) == Err(LoadError::TooShort),
    );
    buffer[20] ^= 1;
    report.check(
        "Beschädigte Daten werden abgelehnt (Prüfsumme)",
        load_parameters(&mut copy, &buffer) == Err(LoadError::Corrupted),
    );

    // 3. Das Perceptron: einmal aus der Konstante, einmal gelernt.
    report.check(
        "Perceptron aus einer Konstante rechnet UND",
        AND.predict(&[1.0, 1.0]) == 1.0 && AND.predict(&[1.0, 0.0]) == 0.0,
    );
    let mut or = Perceptron::<2>::new([0.0, 0.0], 0.0);
    report.check(
        "Perceptron lernt ODER",
        or.fit(&logic::OR, 100, 0.1).is_some(),
    );
    let mut xor = Perceptron::<2>::new([0.0, 0.0], 0.0);
    report.check(
        "Perceptron scheitert an XOR (nicht linear trennbar)",
        xor.fit(&logic::XOR, 100, 0.1).is_none(),
    );

    // 4. Was hat das gekostet?
    let flash = &raw const _flash_end as usize;
    println!(
        "Flash belegt: {flash} Bytes, Stack benutzt: {} Bytes",
        stack_used()
    );

    if report.failed == 0 {
        println!("ALLES OK");
    } else {
        println!("FEHLER: {} Prüfungen sind fehlgeschlagen", report.failed);
    }
    report.failed == 0
}
