//! Polarization: Jones calculus.

/// A Jones vector representing the polarization state of light.
#[derive(Debug, Clone, Copy)]
pub struct JonesVector {
    pub ex: (f64, f64),
    pub ey: (f64, f64),
}

impl JonesVector {
    #[must_use]
    pub const fn new(ex: (f64, f64), ey: (f64, f64)) -> Self {
        Self { ex, ey }
    }

    /// Horizontal linear polarization.
    #[must_use]
    pub const fn horizontal() -> Self {
        Self::new((1.0, 0.0), (0.0, 0.0))
    }

    /// Vertical linear polarization.
    #[must_use]
    pub const fn vertical() -> Self {
        Self::new((0.0, 0.0), (1.0, 0.0))
    }

    /// +45 degree linear polarization.
    #[must_use]
    pub fn diagonal() -> Self {
        let v = 1.0 / core::f64::consts::SQRT_2;
        Self::new((v, 0.0), (v, 0.0))
    }

    /// Right circular polarization.
    #[must_use]
    pub fn right_circular() -> Self {
        let v = 1.0 / core::f64::consts::SQRT_2;
        Self::new((v, 0.0), (0.0, -v))
    }

    /// Left circular polarization.
    #[must_use]
    pub fn left_circular() -> Self {
        let v = 1.0 / core::f64::consts::SQRT_2;
        Self::new((v, 0.0), (0.0, v))
    }

    /// Intensity: |Ex|^2 + |Ey|^2.
    #[must_use]
    pub fn intensity(self) -> f64 {
        self.ey.1.mul_add(
            self.ey.1,
            self.ey.0.mul_add(
                self.ey.0,
                self.ex.0.mul_add(self.ex.0, self.ex.1 * self.ex.1),
            ),
        )
    }
}

/// A 2x2 Jones matrix for optical elements.
#[derive(Debug, Clone, Copy)]
pub struct JonesMatrix {
    pub m00: (f64, f64),
    pub m01: (f64, f64),
    pub m10: (f64, f64),
    pub m11: (f64, f64),
}

/// Complex multiplication helper.
#[must_use]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0.mul_add(b.0, -(a.1 * b.1)), a.0.mul_add(b.1, a.1 * b.0))
}

/// Complex addition helper.
#[must_use]
fn cadd(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}

impl JonesMatrix {
    #[must_use]
    pub const fn new(m00: (f64, f64), m01: (f64, f64), m10: (f64, f64), m11: (f64, f64)) -> Self {
        Self { m00, m01, m10, m11 }
    }

    /// Horizontal linear polarizer.
    #[must_use]
    pub const fn polarizer_h() -> Self {
        Self::new((1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    }

    /// Vertical linear polarizer.
    #[must_use]
    pub const fn polarizer_v() -> Self {
        Self::new((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    }

    /// Linear polarizer at angle `theta`.
    #[must_use]
    pub fn polarizer(theta: f64) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        Self::new((c * c, 0.0), (c * s, 0.0), (c * s, 0.0), (s * s, 0.0))
    }

    /// Half-wave plate with fast axis at angle `theta`.
    #[must_use]
    pub fn half_wave_plate(theta: f64) -> Self {
        let c2 = (2.0 * theta).cos();
        let s2 = (2.0 * theta).sin();
        Self::new((c2, 0.0), (s2, 0.0), (s2, 0.0), (-c2, 0.0))
    }

    /// Quarter-wave plate with fast axis at angle `theta`.
    #[must_use]
    pub fn quarter_wave_plate(theta: f64) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        let cc = c * c;
        let ss = s * s;
        let cs = c * s;
        Self::new((cc, ss), (cs, -cs), (cs, -cs), (ss, cc))
    }

    /// General wave plate with retardance `phi` and fast axis at angle `theta`.
    #[must_use]
    pub fn wave_plate(theta: f64, phi: f64) -> Self {
        let c = theta.cos();
        let _s = theta.sin();
        let cp = (phi / 2.0).cos();
        let sp = (phi / 2.0).sin();
        Self::new(
            (
                (sp * (2.0 * c).mul_add(c, -1.0)).mul_add(0.0_f64.cos(), cp),
                sp * (2.0 * c).mul_add(c, -1.0).sin().copysign(1.0) * 0.0,
            ),
            (0.0, 0.0),
            (0.0, 0.0),
            (cp, 0.0),
        )
    }

    /// Apply this Jones matrix to a Jones vector.
    #[must_use]
    pub fn apply(self, v: JonesVector) -> JonesVector {
        JonesVector {
            ex: cadd(cmul(self.m00, v.ex), cmul(self.m01, v.ey)),
            ey: cadd(cmul(self.m10, v.ex), cmul(self.m11, v.ey)),
        }
    }

    /// Multiply two Jones matrices.
    #[must_use]
    pub fn matmul(self, other: Self) -> Self {
        Self::new(
            cadd(cmul(self.m00, other.m00), cmul(self.m01, other.m10)),
            cadd(cmul(self.m00, other.m01), cmul(self.m01, other.m11)),
            cadd(cmul(self.m10, other.m00), cmul(self.m11, other.m10)),
            cadd(cmul(self.m10, other.m01), cmul(self.m11, other.m11)),
        )
    }
}
