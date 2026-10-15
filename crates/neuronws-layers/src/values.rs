use crate::scalar::Scalar;

/// Eine Gruppe von Zahlen fester Größe, die man der Reihe nach durchnummerieren kann.
///
/// Damit lassen sich Eingaben, Ausgaben und Parameter einheitlich behandeln, egal ob es ein
/// Vektor (`[T; N]`), eine Matrix (`[[T; IN]; OUT]`), mehrere Teile zusammen (`(A, B)`) oder gar
/// nichts (`()`) ist. Die Nummerierung ist fest: Matrizen zeilenweise, Paare erst der linke, dann
/// der rechte Teil.
pub trait Values<T: Scalar>: Copy {
    /// Wie viele Zahlen es sind.
    const LEN: usize;

    fn zeros() -> Self;

    /// Die Zahl mit der Nummer `index`.
    ///
    /// # Panics
    /// Wenn `index >= LEN`.
    fn value(&self, index: usize) -> T;

    /// # Panics
    /// Wenn `index >= LEN`.
    fn value_mut(&mut self, index: usize) -> &mut T;
}

impl<T: Scalar> Values<T> for () {
    const LEN: usize = 0;

    fn zeros() -> Self {}

    fn value(&self, index: usize) -> T {
        panic!("Index {index} außerhalb von 0 Werten")
    }

    fn value_mut(&mut self, index: usize) -> &mut T {
        panic!("Index {index} außerhalb von 0 Werten")
    }
}

/// Eine einzelne Zahl ist eine Gruppe aus einem Wert. Das ist der Boden der Rekursion unten.
impl<T: Scalar> Values<T> for T {
    const LEN: usize = 1;

    fn zeros() -> Self {
        T::ZERO
    }

    fn value(&self, index: usize) -> T {
        assert!(index == 0, "Index {index} außerhalb von 1 Wert");
        *self
    }

    fn value_mut(&mut self, index: usize) -> &mut T {
        assert!(index == 0, "Index {index} außerhalb von 1 Wert");
        self
    }
}

/// Ein Array aus Gruppen ist selbst eine Gruppe, beliebig tief: `[T; N]` ist ein Vektor,
/// `[[T; C]; R]` eine Matrix (zeilenweise), `[[[T; W]; H]; K]` ein Bild mit `K` Kanälen und so fort.
impl<T: Scalar, V: Values<T>, const N: usize> Values<T> for [V; N] {
    const LEN: usize = N * V::LEN;

    fn zeros() -> Self {
        [V::zeros(); N]
    }

    fn value(&self, index: usize) -> T {
        assert!(
            index < Self::LEN,
            "Index {index} außerhalb von {} Werten",
            Self::LEN
        );
        self[index / V::LEN].value(index % V::LEN)
    }

    fn value_mut(&mut self, index: usize) -> &mut T {
        assert!(
            index < Self::LEN,
            "Index {index} außerhalb von {} Werten",
            Self::LEN
        );
        self[index / V::LEN].value_mut(index % V::LEN)
    }
}

impl<T: Scalar, A: Values<T>, B: Values<T>> Values<T> for (A, B) {
    const LEN: usize = A::LEN + B::LEN;

    fn zeros() -> Self {
        (A::zeros(), B::zeros())
    }

    fn value(&self, index: usize) -> T {
        if index < A::LEN {
            self.0.value(index)
        } else {
            self.1.value(index - A::LEN)
        }
    }

    fn value_mut(&mut self, index: usize) -> &mut T {
        if index < A::LEN {
            self.0.value_mut(index)
        } else {
            self.1.value_mut(index - A::LEN)
        }
    }
}

/// Ein Bild mit `C` Kanälen, `H` Zeilen und `W` Spalten: `image[kanal][zeile][spalte]`.
///
/// Nur eine lesbare Abkürzung für `[[[T; W]; H]; C]`.
pub type Image<const C: usize, const H: usize, const W: usize, T = f32> = [[[T; W]; H]; C];
