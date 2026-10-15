//! Gewichte speichern und laden, ohne `std` und ohne `alloc`.
//!
//! Zwei Wege, die beide mit festen Puffern auskommen:
//!
//! | Weg | Ausgabe | Gedacht für |
//! |---|---|---|
//! | [`save_parameters`] / [`load_parameters`] | Bytes in einem `[u8]`-Puffer | Flash, EEPROM, Datei, Funkübertragung |
//! | [`write_rust_const`] | Text einer Rust-Konstante über `core::fmt::Write` | fest in das Programm einkompilieren |
//!
//! Gespeichert werden die **lernbaren Parameter** (alles, was `Trainable` besucht), in dieser
//! Reihenfolge. Das Netz selbst, also seine Form, steckt im Typ und im Programm. Geladen wird
//! immer in ein Netz **desselben Typs**. Die Daten prüfen das, soweit sie können (Zahlenart und
//! Anzahl der Parameter).
//!
//! # Das Byte-Format
//!
//! ```text
//! Bytes  0..4    "NWS1"                      Kennung und Version
//! Byte   4       4 oder 8                    Bytes pro Zahl (f32 oder f64)
//! Bytes  5..8    0 0 0                       reserviert
//! Bytes  8..12   Anzahl der Zahlen, u32      niedrigstes Byte zuerst
//! danach         die Zahlen, je 4 oder 8 Bytes, niedrigstes Byte zuerst
//! zuletzt        CRC-32 über alles davor, u32, niedrigstes Byte zuerst
//! ```
//!
//! Die Prüfsumme fängt kaputte Daten und halb geschriebene Flash-Seiten ab. Geladen wird erst,
//! wenn alles stimmt: Schlägt das Laden fehl, bleibt das Netz **unverändert**.
//!
//! # Was nicht dabei ist
//!
//! Zustand, der nicht gelernt wird, gehört nicht zu `parameters()` und deshalb auch nicht in die
//! Daten: die laufende Statistik einer `BatchNorm`, Mittelwert und Streuung einer `Standardize`.
//! Wer sie braucht, sichert sie getrennt mit [`encode_slice`] und stellt sie beim Aufbau des Netzes
//! wieder her (`with_statistics`, `from_statistics`).
//!
//! ```
//! use neuronws_layers::dense_layer::DenseLayer;
//! use neuronws_layers::persist::{byte_len, load_parameters, save_parameters};
//! use neuronws_layers::traits::ForwardPath;
//!
//! type Net = DenseLayer<2, 1>;
//! let trained = Net::new([[0.5, -1.5]], [0.25]);
//!
//! // Speichern: ein Puffer in der richtigen Größe, ganz ohne Heap.
//! let mut buffer = [0_u8; byte_len::<Net>()];
//! let written = save_parameters(&trained, &mut buffer).unwrap();
//! assert_eq!(written, buffer.len());
//!
//! // Laden: in ein neues Netz derselben Form.
//! let mut restored = Net::new([[0.0, 0.0]], [0.0]);
//! load_parameters(&mut restored, &buffer).unwrap();
//! assert_eq!(restored.forward(&[1.0, 1.0]), trained.forward(&[1.0, 1.0]));
//! ```

use core::fmt;

use crate::scalar::Scalar;
use crate::traits::Trainable;

/// Die Kennung am Anfang der Daten.
const MAGIC: [u8; 4] = *b"NWS1";
/// Kennung (4), Bytes pro Zahl (1), reserviert (3), Anzahl (4).
const HEADER_LEN: usize = 12;
const CHECKSUM_LEN: usize = 4;

/// Wie viele Bytes `count` Zahlen vom Typ `T` im Format oben belegen.
pub const fn encoded_len<T: Scalar>(count: usize) -> usize {
    HEADER_LEN + count * T::BYTES + CHECKSUM_LEN
}

/// Die Größe in Bytes, die [`save_parameters`] für das Modell `M` braucht.
///
/// Eine `const fn`, die sich für feste Puffer eignet, ohne dass man die Parameter zählen muss:
/// `[0_u8; byte_len::<Net>()]`.
pub const fn byte_len<M: Trainable>() -> usize {
    encoded_len::<M::Scalar>(M::PARAMETER_COUNT)
}

/// Der Puffer zum Speichern war zu klein.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferTooSmall {
    /// So viele Bytes braucht es.
    pub needed: usize,
    /// So viele Bytes hat der Puffer.
    pub available: usize,
}

impl fmt::Display for BufferTooSmall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Der Puffer ist mit {} Bytes zu klein, es braucht {}",
            self.available, self.needed
        )
    }
}

impl core::error::Error for BufferTooSmall {}

/// Warum sich Daten nicht in ein Netz laden lassen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    /// Weniger Bytes, als Kopf und Länge verlangen.
    TooShort,
    /// Die Daten beginnen nicht mit der Kennung `NWS1`.
    NotNeuronws,
    /// Gespeichert mit einer anderen Zahlenart (zum Beispiel `f64` statt `f32`).
    WrongScalar {
        /// Bytes pro Zahl in den Daten.
        stored: usize,
        /// Bytes pro Zahl im Netz.
        expected: usize,
    },
    /// Eine andere Zahl von Parametern: Die Daten gehören zu einem anderen Netz.
    WrongCount {
        /// Parameter in den Daten.
        stored: usize,
        /// Parameter im Netz.
        expected: usize,
    },
    /// Die Prüfsumme stimmt nicht: Die Daten sind beschädigt.
    Corrupted,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => write!(f, "Die Daten sind kürzer, als sie sein müssten"),
            Self::NotNeuronws => write!(f, "Die Daten beginnen nicht mit der Kennung NWS1"),
            Self::WrongScalar { stored, expected } => write!(
                f,
                "Die Daten haben {stored} Bytes pro Zahl, das Netz rechnet mit {expected}"
            ),
            Self::WrongCount { stored, expected } => write!(
                f,
                "Die Daten haben {stored} Parameter, das Netz hat {expected}"
            ),
            Self::Corrupted => write!(f, "Die Prüfsumme stimmt nicht, die Daten sind beschädigt"),
        }
    }
}

impl core::error::Error for LoadError {}

/// CRC-32 (IEEE 802.3, wie in zip und PNG), bitweise gerechnet: klein und ohne Tabelle.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Schreibt Kopf und Anzahl in `out`.
fn write_header<T: Scalar>(out: &mut [u8], count: usize) -> Result<(), BufferTooSmall> {
    let needed = encoded_len::<T>(count);
    if out.len() < needed {
        return Err(BufferTooSmall {
            needed,
            available: out.len(),
        });
    }
    assert!(
        u32::try_from(count).is_ok(),
        "mehr als u32::MAX Zahlen lassen sich nicht speichern"
    );
    out[..4].copy_from_slice(&MAGIC);
    out[4] = T::BYTES as u8;
    out[5..8].fill(0);
    out[8..HEADER_LEN].copy_from_slice(&(count as u32).to_le_bytes());
    Ok(())
}

/// Hängt die Prüfsumme an die ersten `len` Bytes und liefert die Gesamtlänge.
fn write_checksum(out: &mut [u8], len: usize) -> usize {
    let crc = crc32(&out[..len]);
    out[len..len + CHECKSUM_LEN].copy_from_slice(&crc.to_le_bytes());
    len + CHECKSUM_LEN
}

/// Prüft Kennung, Zahlenart, Prüfsumme und liefert die Anzahl der Zahlen.
fn read_header<T: Scalar>(bytes: &[u8]) -> Result<usize, LoadError> {
    if bytes.len() < HEADER_LEN + CHECKSUM_LEN {
        return Err(LoadError::TooShort);
    }
    if bytes[..4] != MAGIC {
        return Err(LoadError::NotNeuronws);
    }
    let stored = usize::from(bytes[4]);
    if stored != T::BYTES {
        return Err(LoadError::WrongScalar {
            stored,
            expected: T::BYTES,
        });
    }
    let mut count_bytes = [0; 4];
    count_bytes.copy_from_slice(&bytes[8..HEADER_LEN]);
    let count = u32::from_le_bytes(count_bytes) as usize;

    let total = encoded_len::<T>(count);
    if bytes.len() < total {
        return Err(LoadError::TooShort);
    }
    let mut crc_bytes = [0; 4];
    crc_bytes.copy_from_slice(&bytes[total - CHECKSUM_LEN..total]);
    if crc32(&bytes[..total - CHECKSUM_LEN]) != u32::from_le_bytes(crc_bytes) {
        return Err(LoadError::Corrupted);
    }
    Ok(count)
}

/// Schreibt alle lernbaren Parameter von `model` in `out` und liefert die Zahl der geschriebenen
/// Bytes (die ersten `n` Bytes von `out` sind dann die Daten).
///
/// Die Größe liefert [`byte_len`]. Ist der Puffer kleiner, bleibt er unberührt.
///
/// # Errors
/// [`BufferTooSmall`], wenn `out` zu kurz ist.
pub fn save_parameters<M: Trainable>(model: &M, out: &mut [u8]) -> Result<usize, BufferTooSmall> {
    let count = M::PARAMETER_COUNT;
    write_header::<M::Scalar>(out, count)?;
    let mut offset = HEADER_LEN;
    model.inspect_parameters(&mut |parameter, _| {
        parameter.write_le(&mut out[offset..]);
        offset += <M::Scalar as Scalar>::BYTES;
    });
    Ok(write_checksum(out, offset))
}

/// Liest Parameter, die [`save_parameters`] geschrieben hat, in `model`.
///
/// Erst wird alles geprüft (Kennung, Zahlenart, Anzahl, Prüfsumme), dann geladen. Bei einem Fehler
/// bleibt das Netz unverändert. Gradienten und Zwischenspeicher des Netzes werden nicht angefasst.
///
/// # Errors
/// [`LoadError`], wenn die Daten nicht zu diesem Netz passen oder beschädigt sind.
pub fn load_parameters<M: Trainable>(model: &mut M, bytes: &[u8]) -> Result<(), LoadError> {
    let count = read_header::<M::Scalar>(bytes)?;
    if count != M::PARAMETER_COUNT {
        return Err(LoadError::WrongCount {
            stored: count,
            expected: M::PARAMETER_COUNT,
        });
    }
    let mut offset = HEADER_LEN;
    model.visit_parameters(&mut |parameter, _| {
        *parameter = <M::Scalar as Scalar>::read_le(&bytes[offset..]);
        offset += <M::Scalar as Scalar>::BYTES;
    });
    Ok(())
}

/// Wie [`save_parameters`], aber für eine beliebige Liste von Zahlen. Damit sichert man alles, was
/// nicht zu `parameters()` gehört, zum Beispiel `BatchNorm::running_mean`.
///
/// # Errors
/// [`BufferTooSmall`], wenn `out` zu kurz ist.
pub fn encode_slice<T: Scalar>(values: &[T], out: &mut [u8]) -> Result<usize, BufferTooSmall> {
    write_header::<T>(out, values.len())?;
    let mut offset = HEADER_LEN;
    for &value in values {
        value.write_le(&mut out[offset..]);
        offset += T::BYTES;
    }
    Ok(write_checksum(out, offset))
}

/// Gegenstück zu [`encode_slice`]: liest die Zahlen in `out` und liefert, wie viele es sind.
///
/// `out` muss genau so lang sein wie die gespeicherte Liste (oder länger, dann bleibt der Rest
/// unberührt).
///
/// # Errors
/// [`LoadError`], wenn die Daten beschädigt sind, nicht zur Zahlenart passen oder mehr Zahlen
/// enthalten, als in `out` passen (dann `WrongCount`).
pub fn decode_slice<T: Scalar>(bytes: &[u8], out: &mut [T]) -> Result<usize, LoadError> {
    let count = read_header::<T>(bytes)?;
    if count > out.len() {
        return Err(LoadError::WrongCount {
            stored: count,
            expected: out.len(),
        });
    }
    for (index, slot) in out.iter_mut().take(count).enumerate() {
        *slot = T::read_le(&bytes[HEADER_LEN + index * T::BYTES..]);
    }
    Ok(count)
}

/// Setzt die Parameter von `model` aus einer Liste von Zahlen, in der Reihenfolge von
/// `visit_parameters`. Das passt zu einer Konstanten, die [`write_rust_const`] ausgegeben hat:
///
/// ```
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::persist::parameters_from_slice;
/// use neuronws_layers::traits::ForwardPath;
///
/// // So steht die Ausgabe von `write_rust_const` im Programm: erst die Gewichte, dann der Bias.
/// const WEIGHTS: [f32; 3] = [0.5, -1.5, 0.25];
///
/// let mut net = DenseLayer::<2, 1>::new([[0.0, 0.0]], [0.0]);
/// parameters_from_slice(&mut net, &WEIGHTS).unwrap();
/// assert_eq!(net.forward(&[2.0, 1.0]), [0.5 * 2.0 - 1.5 + 0.25]);
/// ```
///
/// # Errors
/// [`LoadError::WrongCount`], wenn die Liste nicht genau so viele Zahlen hat wie das Netz
/// Parameter. Dann bleibt das Netz unverändert.
pub fn parameters_from_slice<M: Trainable>(
    model: &mut M,
    values: &[M::Scalar],
) -> Result<(), LoadError> {
    if values.len() != M::PARAMETER_COUNT {
        return Err(LoadError::WrongCount {
            stored: values.len(),
            expected: M::PARAMETER_COUNT,
        });
    }
    let mut index = 0;
    model.visit_parameters(&mut |parameter, _| {
        *parameter = values[index];
        index += 1;
    });
    Ok(())
}

/// Kopiert die Parameter von `model` in eine Liste, in der Reihenfolge von `visit_parameters`.
///
/// # Errors
/// [`BufferTooSmall`], wenn `out` kürzer ist als die Zahl der Parameter (gemessen in Zahlen, nicht
/// in Bytes). Dann bleibt `out` unverändert.
pub fn parameters_to_slice<M: Trainable>(
    model: &M,
    out: &mut [M::Scalar],
) -> Result<usize, BufferTooSmall> {
    if out.len() < M::PARAMETER_COUNT {
        return Err(BufferTooSmall {
            needed: M::PARAMETER_COUNT,
            available: out.len(),
        });
    }
    let mut index = 0;
    model.inspect_parameters(&mut |parameter, _| {
        out[index] = parameter;
        index += 1;
    });
    Ok(M::PARAMETER_COUNT)
}

/// Schreibt die Parameter als Rust-Konstante in `out` (alles, was `core::fmt::Write` kann, zum
/// Beispiel ein `String` oder ein eigener Puffer, der auf eine serielle Schnittstelle schreibt):
///
/// ```text
/// pub const WEIGHTS: [f32; 3] = [
///     0.5, -1.5, 0.25,
/// ];
/// ```
///
/// Das Ergebnis kopiert man in den Quelltext des Programms und lädt es mit
/// [`parameters_from_slice`]. So landen die Gewichte im Flash des Mikrocontrollers, ohne dass etwas
/// zur Laufzeit gelesen oder geparst werden muss. Zahlen werden so geschrieben, dass sie sich
/// verlustfrei wieder einlesen lassen (`NaN` und die Unendlichen als `f32::NAN` und so weiter).
///
/// ```
/// use neuronws_layers::dense_layer::DenseLayer;
/// use neuronws_layers::persist::write_rust_const;
///
/// let net = DenseLayer::<2, 1>::new([[0.5, -1.5]], [0.25]);
/// let mut text = String::new();
/// write_rust_const(&net, "WEIGHTS", &mut text).unwrap();
/// assert!(text.starts_with("pub const WEIGHTS: [f32; 3] = ["));
/// assert!(text.contains("0.5, -1.5, 0.25"));
/// ```
///
/// # Errors
/// Was der Schreiber meldet.
pub fn write_rust_const<M, W>(model: &M, name: &str, out: &mut W) -> fmt::Result
where
    M: Trainable,
    M::Scalar: fmt::Debug,
    W: fmt::Write,
{
    /// So viele Zahlen pro Zeile.
    const PER_LINE: usize = 8;

    let type_name = if <M::Scalar as Scalar>::BYTES == 4 {
        "f32"
    } else {
        "f64"
    };
    writeln!(
        out,
        "pub const {name}: [{type_name}; {}] = [",
        M::PARAMETER_COUNT
    )?;

    let mut result = Ok(());
    let mut index = 0;
    model.inspect_parameters(&mut |parameter, _| {
        if result.is_err() {
            return;
        }
        let separator = match index % PER_LINE {
            0 => "    ",
            _ => " ",
        };
        let last_in_line = index % PER_LINE == PER_LINE - 1 || index + 1 == M::PARAMETER_COUNT;
        result = write_literal(out, separator, type_name, parameter, last_in_line);
        index += 1;
    });
    result?;
    writeln!(out, "];")
}

/// Eine Zahl als gültiges Rust-Literal, mit Komma und, am Zeilenende, einem Zeilenumbruch.
fn write_literal<T: fmt::Debug + Scalar, W: fmt::Write>(
    out: &mut W,
    separator: &str,
    type_name: &str,
    value: T,
    end_of_line: bool,
) -> fmt::Result {
    let as_f64 = value.to_f64();
    out.write_str(separator)?;
    if as_f64.is_nan() {
        write!(out, "{type_name}::NAN")?;
    } else if as_f64 == f64::INFINITY {
        write!(out, "{type_name}::INFINITY")?;
    } else if as_f64 == f64::NEG_INFINITY {
        write!(out, "{type_name}::NEG_INFINITY")?;
    } else {
        write!(out, "{value:?}")?;
    }
    out.write_str(if end_of_line { ",\n" } else { "," })
}
