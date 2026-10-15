/// Ein kleiner Zufallsgenerator (xorshift64*) ohne Abhängigkeiten.
///
/// Er ist deterministisch: derselbe Startwert (`seed`) liefert immer dieselbe Folge.
/// Das ist für reproduzierbare Experimente gewollt. Für Kryptografie taugt er nicht.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub const fn new(seed: u64) -> Self {
        // SplitMix64 mischt auch kleine Startwerte (0, 1, 2, ...) gut durch.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // xorshift bleibt bei 0 für immer bei 0
        Self {
            state: if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z },
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// Gleichverteilt in `[0, 1)`. Die oberen 24 Bit passen genau in die Mantisse eines `f32`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Gleichverteilt in `[low, high)`.
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }
}
