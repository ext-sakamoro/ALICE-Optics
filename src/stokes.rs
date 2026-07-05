//! Polarization: Stokes parameters and Mueller matrices.

use core::f64::consts::PI;

/// Stokes vector `[S0, S1, S2, S3]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StokesVector {
    pub s0: f64,
    pub s1: f64,
    pub s2: f64,
    pub s3: f64,
}

impl StokesVector {
    #[must_use]
    pub const fn new(s0: f64, s1: f64, s2: f64, s3: f64) -> Self {
        Self { s0, s1, s2, s3 }
    }

    /// Unpolarized light.
    #[must_use]
    pub const fn unpolarized(intensity: f64) -> Self {
        Self::new(intensity, 0.0, 0.0, 0.0)
    }

    /// Horizontal linear polarization.
    #[must_use]
    pub const fn horizontal(intensity: f64) -> Self {
        Self::new(intensity, intensity, 0.0, 0.0)
    }

    /// Vertical linear polarization.
    #[must_use]
    pub const fn vertical(intensity: f64) -> Self {
        Self::new(intensity, -intensity, 0.0, 0.0)
    }

    /// Right circular polarization.
    #[must_use]
    pub const fn right_circular(intensity: f64) -> Self {
        Self::new(intensity, 0.0, 0.0, intensity)
    }

    /// Left circular polarization.
    #[must_use]
    pub const fn left_circular(intensity: f64) -> Self {
        Self::new(intensity, 0.0, 0.0, -intensity)
    }

    /// Degree of polarization.
    #[must_use]
    pub fn degree_of_polarization(self) -> f64 {
        if self.s0.abs() < 1e-15 {
            return 0.0;
        }
        self.s3
            .mul_add(self.s3, self.s1.mul_add(self.s1, self.s2 * self.s2))
            .sqrt()
            / self.s0
    }

    /// Polarization ellipse orientation angle.
    #[must_use]
    pub fn orientation_angle(self) -> f64 {
        0.5 * self.s2.atan2(self.s1)
    }

    /// Polarization ellipticity angle.
    #[must_use]
    pub fn ellipticity_angle(self) -> f64 {
        let denom = self.s1.hypot(self.s2);
        if denom < 1e-15 {
            if self.s3 > 0.0 {
                return PI / 4.0;
            } else if self.s3 < 0.0 {
                return -PI / 4.0;
            }
            return 0.0;
        }
        0.5 * (self.s3 / denom).atan()
    }
}

/// Mueller matrix (4x4) for Stokes vector transformations.
#[derive(Debug, Clone, Copy)]
pub struct MuellerMatrix {
    pub m: [[f64; 4]; 4],
}

impl MuellerMatrix {
    #[must_use]
    pub const fn new(m: [[f64; 4]; 4]) -> Self {
        Self { m }
    }

    /// Identity Mueller matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Horizontal linear polarizer Mueller matrix.
    #[must_use]
    pub const fn polarizer_h() -> Self {
        Self::new([
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }

    /// Vertical linear polarizer Mueller matrix.
    #[must_use]
    pub const fn polarizer_v() -> Self {
        Self::new([
            [0.5, -0.5, 0.0, 0.0],
            [-0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }

    /// Apply Mueller matrix to Stokes vector.
    #[must_use]
    pub fn apply(self, s: StokesVector) -> StokesVector {
        let sv = [s.s0, s.s1, s.s2, s.s3];
        let mut out = [0.0; 4];
        for (i, out_val) in out.iter_mut().enumerate() {
            for (j, sv_val) in sv.iter().enumerate() {
                *out_val += self.m[i][j] * sv_val;
            }
        }
        StokesVector::new(out[0], out[1], out[2], out[3])
    }
}
