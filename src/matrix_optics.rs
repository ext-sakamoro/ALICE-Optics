//! ABCD ray transfer matrix optics + paraxial spherical surface tracing.

/// 2x2 ray transfer matrix `[[a, b], [c, d]]`.
#[derive(Debug, Clone, Copy)]
pub struct Matrix2x2 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl Matrix2x2 {
    #[must_use]
    pub const fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    #[must_use]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0)
    }

    /// Free-space propagation matrix.
    #[must_use]
    pub const fn propagation(d: f64) -> Self {
        Self::new(1.0, d, 0.0, 1.0)
    }

    /// Thin lens refraction matrix.
    #[must_use]
    pub fn thin_lens(f: f64) -> Self {
        Self::new(1.0, 0.0, -1.0 / f, 1.0)
    }

    /// Flat interface refraction matrix.
    #[must_use]
    pub const fn flat_refraction(n1: f64, n2: f64) -> Self {
        Self::new(1.0, 0.0, 0.0, n1 / n2)
    }

    /// Curved interface refraction matrix.
    #[must_use]
    pub fn curved_refraction(n1: f64, n2: f64, r: f64) -> Self {
        Self::new(1.0, 0.0, (n1 - n2) / (n2 * r), n1 / n2)
    }

    /// Multiply two matrices: self * other.
    #[must_use]
    pub fn matmul(self, other: Self) -> Self {
        Self::new(
            self.a.mul_add(other.a, self.b * other.c),
            self.a.mul_add(other.b, self.b * other.d),
            self.c.mul_add(other.a, self.d * other.c),
            self.c.mul_add(other.b, self.d * other.d),
        )
    }

    /// Apply matrix to ray vector `[y, theta]`.
    #[must_use]
    pub fn apply(self, y: f64, theta: f64) -> (f64, f64) {
        (
            self.a.mul_add(y, self.b * theta),
            self.c.mul_add(y, self.d * theta),
        )
    }

    /// Determinant.
    #[must_use]
    pub fn det(self) -> f64 {
        self.a.mul_add(self.d, -(self.b * self.c))
    }
}

/// Compose a sequence of ABCD matrices (applied left to right in optical order).
/// The first element in the slice is the first optical element the ray encounters.
#[must_use]
pub fn compose_matrices(matrices: &[Matrix2x2]) -> Matrix2x2 {
    matrices
        .iter()
        .rev()
        .copied()
        .fold(Matrix2x2::identity(), |acc, m| m.matmul(acc))
}

/// A spherical optical surface.
#[derive(Debug, Clone, Copy)]
pub struct SphericalSurface {
    /// Z-position along the optical axis.
    pub z: f64,
    /// Radius of curvature (positive = center to the right).
    pub radius: f64,
    /// Refractive index after this surface.
    pub n_after: f64,
}

/// Trace a paraxial ray through a sequence of spherical surfaces.
/// Returns `(final_height, final_angle)`.
/// `n_before` is the refractive index before the first surface.
#[must_use]
pub fn trace_paraxial(
    surfaces: &[SphericalSurface],
    n_before: f64,
    y0: f64,
    u0: f64,
) -> (f64, f64) {
    let mut y = y0;
    let mut u = u0;
    let mut n = n_before;
    let mut z_prev = 0.0;

    for surf in surfaces {
        let d = surf.z - z_prev;
        y += d * u;
        let n_next = surf.n_after;
        u = n.mul_add(u, (n - n_next) * y / surf.radius) / n_next;
        n = n_next;
        z_prev = surf.z;
    }

    (y, u)
}
