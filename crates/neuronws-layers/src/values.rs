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

impl<T: Scalar, const N: usize> Values<T> for [T; N] {
    const LEN: usize = N;

    fn zeros() -> Self {
        [T::ZERO; N]
    }

    fn value(&self, index: usize) -> T {
        self[index]
    }

    fn value_mut(&mut self, index: usize) -> &mut T {
        &mut self[index]
    }
}

impl<T: Scalar, const COLUMNS: usize, const ROWS: usize> Values<T> for [[T; COLUMNS]; ROWS] {
    const LEN: usize = COLUMNS * ROWS;

    fn zeros() -> Self {
        [[T::ZERO; COLUMNS]; ROWS]
    }

    fn value(&self, index: usize) -> T {
        assert!(
            index < Self::LEN,
            "Index {index} außerhalb von {} Werten",
            Self::LEN
        );
        self[index / COLUMNS][index % COLUMNS]
    }

    fn value_mut(&mut self, index: usize) -> &mut T {
        assert!(
            index < Self::LEN,
            "Index {index} außerhalb von {} Werten",
            Self::LEN
        );
        &mut self[index / COLUMNS][index % COLUMNS]
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
