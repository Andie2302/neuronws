//! Gewichte speichern und laden: Bytes mit Prüfsumme und Rust-Konstanten.

use neuronws_layers::activation::Relu;
use neuronws_layers::activation_layer::ActivationLayer;
use neuronws_layers::chain::{Chain, LayerExt};
use neuronws_layers::dense_layer::DenseLayer;
use neuronws_layers::persist::{
    BufferTooSmall, LoadError, byte_len, decode_slice, encode_slice, encoded_len, load_parameters,
    parameters_from_slice, parameters_to_slice, save_parameters, write_rust_const,
};
use neuronws_layers::traits::{ForwardPath, Trainable};

type Net = Chain<Chain<DenseLayer<2, 3>, ActivationLayer<3, Relu>>, DenseLayer<3, 1>>;

fn net(seed: f32) -> Net {
    DenseLayer::new(
        [[0.1 + seed, -0.2], [0.3, 0.4 * seed], [-0.5, 0.6]],
        [0.01, -0.02, 0.03 + seed],
    )
    .then(ActivationLayer::<3, _>::new(Relu))
    .then(DenseLayer::new([[0.7, -0.8 * seed, 0.9]], [0.05]))
}

/// Ein unabhängig geschriebenes CRC-32 zum Gegenprüfen der Prüfsumme in den Daten.
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

fn saved(model: &Net) -> Vec<u8> {
    let mut bytes = vec![0; byte_len::<Net>()];
    let written = save_parameters(model, &mut bytes).unwrap();
    assert_eq!(written, bytes.len());
    bytes
}

#[test]
fn the_reference_crc_matches_the_known_check_value() {
    // Der übliche Prüfwert für CRC-32: die ASCII-Ziffern 1 bis 9.
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
}

#[test]
fn a_saved_net_loads_back_exactly() {
    let trained = net(1.0);
    let bytes = saved(&trained);

    let mut restored = net(-3.0);
    assert_ne!(restored.parameters(), trained.parameters());
    load_parameters(&mut restored, &bytes).unwrap();

    assert_eq!(restored.parameters(), trained.parameters());
    for input in [[0.0, 0.0], [1.0, -1.0], [0.3, 7.0]] {
        assert_eq!(restored.forward(&input), trained.forward(&input));
    }
}

#[test]
fn it_works_for_f64_and_for_other_layers() {
    use neuronws_layers::conv::Conv2dValidLayer;

    type Cnn = Conv2dValidLayer<1, 2, 4, 4, 3, 3, 2, 2, f64>;
    let kernels = core::array::from_fn(|o| {
        core::array::from_fn(|_| {
            core::array::from_fn(|p| core::array::from_fn(|q| (o + p * 3 + q) as f64 * 0.1 - 0.2))
        })
    });
    let cnn = Cnn::new(kernels, [0.5, -0.25]);
    let mut bytes = vec![0; byte_len::<Cnn>()];
    save_parameters(&cnn, &mut bytes).unwrap();
    assert_eq!(bytes[4], 8, "f64: 8 Bytes pro Zahl");

    let mut restored = Cnn::new([[[[0.0; 3]; 3]]; 2], [0.0; 2]);
    load_parameters(&mut restored, &bytes).unwrap();
    assert_eq!(restored.kernels(), cnn.kernels());
    assert_eq!(restored.bias(), cnn.bias());
}

#[test]
fn the_size_is_known_at_compile_time() {
    const N: usize = byte_len::<Net>();
    // Kopf 12 + 9 + 4 Parameter... : (3·2 + 3) + (1·3 + 1) = 13 Zahlen zu je 4 Bytes + Prüfsumme 4
    assert_eq!(<Net as Trainable>::PARAMETER_COUNT, 13);
    assert_eq!(N, 12 + 13 * 4 + 4);
    let buffer = [0_u8; N];
    assert_eq!(buffer.len(), 68);
    assert_eq!(encoded_len::<f64>(10), 12 + 80 + 4);
}

#[test]
fn the_layout_is_as_documented() {
    let bytes = saved(&net(1.0));
    assert_eq!(&bytes[..4], b"NWS1");
    assert_eq!(bytes[4], 4);
    assert_eq!(&bytes[5..8], &[0, 0, 0]);
    assert_eq!(&bytes[8..12], &13_u32.to_le_bytes());
    // die erste Zahl ist das erste Gewicht, niedrigstes Byte zuerst
    assert_eq!(&bytes[12..16], &(0.1_f32 + 1.0).to_le_bytes());
    // am Ende die Prüfsumme über alles davor
    let (body, crc) = bytes.split_at(bytes.len() - 4);
    assert_eq!(crc, &crc32(body).to_le_bytes());
}

#[test]
fn a_too_small_buffer_is_reported_and_left_alone() {
    let model = net(1.0);
    let mut small = vec![0xAA_u8; byte_len::<Net>() - 1];
    let error = save_parameters(&model, &mut small).unwrap_err();
    assert_eq!(
        error,
        BufferTooSmall {
            needed: byte_len::<Net>(),
            available: byte_len::<Net>() - 1,
        }
    );
    assert!(small.iter().all(|&b| b == 0xAA), "unberührt");
    assert!(error.to_string().contains("zu klein"));
}

#[test]
fn a_bigger_buffer_is_fine_and_trailing_bytes_are_ignored_when_loading() {
    let model = net(1.0);
    let mut big = vec![0xEE_u8; byte_len::<Net>() + 10];
    let written = save_parameters(&model, &mut big).unwrap();
    assert_eq!(written, byte_len::<Net>());

    let mut restored = net(0.0);
    load_parameters(&mut restored, &big).unwrap();
    assert_eq!(restored.parameters(), model.parameters());
}

#[test]
fn truncated_data_is_too_short() {
    let bytes = saved(&net(1.0));
    let mut model = net(0.0);
    assert_eq!(load_parameters(&mut model, &[]), Err(LoadError::TooShort));
    assert_eq!(
        load_parameters(&mut model, &bytes[..10]),
        Err(LoadError::TooShort)
    );
    assert_eq!(
        load_parameters(&mut model, &bytes[..bytes.len() - 1]),
        Err(LoadError::TooShort)
    );
}

#[test]
fn foreign_data_is_recognised() {
    let mut bytes = saved(&net(1.0));
    bytes[0] = b'X';
    assert_eq!(
        load_parameters(&mut net(0.0), &bytes),
        Err(LoadError::NotNeuronws)
    );
}

#[test]
fn another_kind_of_number_is_refused() {
    let double = DenseLayer::<2, 1, f64>::new([[1.0, 2.0]], [3.0]);
    let mut bytes = vec![0; byte_len::<DenseLayer<2, 1, f64>>()];
    save_parameters(&double, &mut bytes).unwrap();

    let mut single = DenseLayer::<2, 1>::new([[0.0; 2]], [0.0]);
    assert_eq!(
        load_parameters(&mut single, &bytes),
        Err(LoadError::WrongScalar {
            stored: 8,
            expected: 4
        })
    );
}

#[test]
fn data_of_a_different_net_is_refused() {
    let bytes = saved(&net(1.0));
    let mut other = DenseLayer::<2, 1>::new([[0.0; 2]], [0.0]);
    let error = load_parameters(&mut other, &bytes).unwrap_err();
    assert_eq!(
        error,
        LoadError::WrongCount {
            stored: 13,
            expected: 3
        }
    );
    assert!(error.to_string().contains("13"));
}

#[test]
fn a_flipped_bit_is_noticed_anywhere() {
    let bytes = saved(&net(1.0));
    // jedes Byte hinter der Kennung und der Zahlenart, einmal ein Bit gekippt
    for position in 5..bytes.len() {
        let mut damaged = bytes.clone();
        damaged[position] ^= 0x10;
        let result = load_parameters(&mut net(0.0), &damaged);
        assert!(result.is_err(), "Byte {position} unbemerkt verändert");
    }
}

#[test]
fn a_failed_load_leaves_the_net_unchanged() {
    let mut bytes = saved(&net(1.0));
    let last_value = bytes.len() - 8;
    bytes[last_value] ^= 0xFF;

    let mut model = net(0.0);
    let before = model.parameters();
    assert_eq!(
        load_parameters(&mut model, &bytes),
        Err(LoadError::Corrupted)
    );
    assert_eq!(model.parameters(), before);
}

#[test]
fn loading_keeps_gradients_and_caches() {
    use neuronws_layers::traits::{BackwardPathMut, ForwardPathMut};

    let mut model = net(0.0);
    model.forward_mut(&[1.0, 2.0]);
    model.backward_mut(&[1.0]);
    let mut grads_before = vec![];
    model.inspect_parameters(&mut |_, g| grads_before.push(g));

    load_parameters(&mut model, &saved(&net(1.0))).unwrap();
    let mut grads_after = vec![];
    model.inspect_parameters(&mut |_, g| grads_after.push(g));
    assert_eq!(grads_before, grads_after);
}

mod slices {
    use super::*;

    #[test]
    fn parameters_go_to_a_slice_and_come_back() {
        let model = net(1.0);
        let mut values = [0.0_f32; 13];
        assert_eq!(parameters_to_slice(&model, &mut values), Ok(13));

        let mut restored = net(-2.0);
        parameters_from_slice(&mut restored, &values).unwrap();
        assert_eq!(restored.parameters(), model.parameters());
    }

    #[test]
    fn the_slice_must_have_the_right_length() {
        let mut model = net(1.0);
        let before = model.parameters();
        assert_eq!(
            parameters_from_slice(&mut model, &[0.0; 12]),
            Err(LoadError::WrongCount {
                stored: 12,
                expected: 13
            })
        );
        assert_eq!(model.parameters(), before, "unverändert");

        let mut short = [0.0_f32; 5];
        assert_eq!(
            parameters_to_slice(&model, &mut short),
            Err(BufferTooSmall {
                needed: 13,
                available: 5
            })
        );
        assert_eq!(short, [0.0; 5]);
    }

    #[test]
    fn arbitrary_numbers_can_be_encoded_and_decoded() {
        // Zum Beispiel die laufende Statistik einer BatchNorm.
        let statistics = [1.5_f32, -0.25, 8.0, f32::MIN_POSITIVE];
        let mut bytes = [0_u8; 12 + 16 + 4];
        assert_eq!(encode_slice(&statistics, &mut bytes), Ok(bytes.len()));

        let mut decoded = [0.0_f32; 4];
        assert_eq!(decode_slice(&bytes, &mut decoded), Ok(4));
        assert_eq!(decoded, statistics);
    }

    #[test]
    fn decoding_checks_the_same_things() {
        let mut bytes = [0_u8; 12 + 8 + 4];
        encode_slice(&[1.0_f32, 2.0], &mut bytes).unwrap();

        // ein zu kleines Ziel
        let mut small = [0.0_f32; 1];
        assert_eq!(
            decode_slice(&bytes, &mut small),
            Err(LoadError::WrongCount {
                stored: 2,
                expected: 1
            })
        );
        // eine andere Zahlenart
        let mut doubles = [0.0_f64; 2];
        assert_eq!(
            decode_slice(&bytes, &mut doubles),
            Err(LoadError::WrongScalar {
                stored: 4,
                expected: 8
            })
        );
        // ein größeres Ziel lässt den Rest unberührt
        let mut bigger = [9.0_f32; 4];
        assert_eq!(decode_slice(&bytes, &mut bigger), Ok(2));
        assert_eq!(bigger, [1.0, 2.0, 9.0, 9.0]);
        // beschädigt
        bytes[13] ^= 1;
        assert_eq!(decode_slice(&bytes, &mut bigger), Err(LoadError::Corrupted));
        // zu kurzer Puffer beim Schreiben
        assert!(encode_slice(&[1.0_f32; 3], &mut [0_u8; 20]).is_err());
    }

    #[test]
    fn an_empty_list_round_trips() {
        let mut bytes = [0_u8; 16];
        assert_eq!(encode_slice::<f32>(&[], &mut bytes), Ok(16));
        assert_eq!(decode_slice::<f32>(&bytes, &mut []), Ok(0));
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn batch_norm_statistics_travel_separately() {
    use neuronws_layers::norm::BatchNorm1dLayer;
    use neuronws_layers::traits::ForwardPathMut;

    type Layer = BatchNorm1dLayer<2>;
    let mut trained = Layer::new();
    for step in 0..50 {
        trained.forward_mut(&[1.0 + step as f32 * 0.1, -2.0 + (step % 3) as f32]);
    }

    // Die Parameter (γ und β) allein reichen nicht: Die Statistik fehlt.
    let mut only_parameters = Layer::new();
    let mut bytes = vec![0; byte_len::<Layer>()];
    save_parameters(&trained, &mut bytes).unwrap();
    load_parameters(&mut only_parameters, &bytes).unwrap();
    assert_ne!(
        only_parameters.forward(&[3.0, 3.0]),
        trained.forward(&[3.0, 3.0])
    );

    // Mit der Statistik als zweite Datei stimmt das Netz wieder.
    let mut stats_bytes = [0_u8; encoded_len::<f32>(4)];
    let stats = [*trained.running_mean(), *trained.running_var()].concat();
    encode_slice(&stats, &mut stats_bytes).unwrap();
    let mut decoded = [0.0_f32; 4];
    decode_slice(&stats_bytes, &mut decoded).unwrap();
    let mut restored =
        Layer::new().with_statistics([decoded[0], decoded[1]], [decoded[2], decoded[3]]);
    load_parameters(&mut restored, &bytes).unwrap();
    assert_eq!(restored.forward(&[3.0, 3.0]), trained.forward(&[3.0, 3.0]));
}

mod rust_const {
    use super::*;
    use core::fmt::{self, Write};

    #[test]
    fn it_writes_a_const_that_the_compiler_would_accept() {
        let model = DenseLayer::<2, 1>::new([[0.5, -1.5]], [0.25]);
        let mut text = String::new();
        write_rust_const(&model, "WEIGHTS", &mut text).unwrap();
        assert_eq!(
            text,
            "pub const WEIGHTS: [f32; 3] = [\n    0.5, -1.5, 0.25,\n];\n"
        );
    }

    #[test]
    fn eight_numbers_per_line() {
        let text = {
            let mut text = String::new();
            write_rust_const(&net(1.0), "NET", &mut text).unwrap();
            text
        };
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "pub const NET: [f32; 13] = [");
        assert_eq!(lines[1].matches(',').count(), 8);
        assert_eq!(lines[2].matches(',').count(), 5);
        assert_eq!(lines[3], "];");
    }

    #[test]
    fn the_numbers_read_back_exactly() {
        let model = net(0.37);
        let mut text = String::new();
        write_rust_const(&model, "NET", &mut text).unwrap();

        let values: Vec<f32> = text
            .lines()
            .skip(1)
            .take_while(|line| *line != "];")
            .flat_map(|line| line.split(','))
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(|token| token.parse().unwrap())
            .collect();
        assert_eq!(values.len(), 13);

        let mut restored = net(-1.0);
        parameters_from_slice(&mut restored, &values).unwrap();
        assert_eq!(restored.parameters(), model.parameters());
    }

    #[test]
    fn f64_gets_its_own_type_and_keeps_all_digits() {
        let model = DenseLayer::<1, 1, f64>::new([[0.1 + 0.2]], [1.0 / 3.0]);
        let mut text = String::new();
        write_rust_const(&model, "W", &mut text).unwrap();
        assert!(text.starts_with("pub const W: [f64; 2] = ["));
        assert!(text.contains("0.30000000000000004"));
        assert!(text.contains("0.3333333333333333"));
    }

    #[test]
    fn special_values_are_valid_rust() {
        let model = DenseLayer::<2, 1>::new([[f32::NAN, f32::INFINITY]], [f32::NEG_INFINITY]);
        let mut text = String::new();
        write_rust_const(&model, "W", &mut text).unwrap();
        assert!(text.contains("f32::NAN"));
        assert!(text.contains("f32::INFINITY"));
        assert!(text.contains("f32::NEG_INFINITY"));
    }

    #[test]
    fn a_net_without_parameters_gives_an_empty_array() {
        let model = ActivationLayer::<3, _>::new(Relu);
        let mut text = String::new();
        write_rust_const(&model, "NONE", &mut text).unwrap();
        assert_eq!(text, "pub const NONE: [f32; 0] = [\n];\n");
    }

    /// Ein Schreiber, der nach `limit` Bytes aufgibt, wie ein voller Puffer.
    struct Limited {
        left: usize,
    }

    impl Write for Limited {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            if s.len() > self.left {
                return Err(fmt::Error);
            }
            self.left -= s.len();
            Ok(())
        }
    }

    #[test]
    fn a_failing_writer_makes_the_call_fail() {
        let model = net(1.0);
        assert!(write_rust_const(&model, "NET", &mut Limited { left: 20 }).is_err());
        assert!(write_rust_const(&model, "NET", &mut Limited { left: 10_000 }).is_ok());
    }
}

#[test]
fn the_errors_are_ordinary_errors() {
    fn is_error<E: core::error::Error>(_: &E) {}
    is_error(&LoadError::Corrupted);
    is_error(&BufferTooSmall {
        needed: 1,
        available: 0,
    });
    for error in [
        LoadError::TooShort,
        LoadError::NotNeuronws,
        LoadError::Corrupted,
        LoadError::WrongScalar {
            stored: 8,
            expected: 4,
        },
        LoadError::WrongCount {
            stored: 1,
            expected: 2,
        },
    ] {
        assert!(!error.to_string().is_empty());
    }
}
