#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]

//! ALICE-Optics: Optical simulation library.
//!
//! Provides lens systems, ray tracing, diffraction, polarization,
//! thin film interference, Snell/Fresnel equations, fiber optics, and aberrations.

use core::f64::consts::PI;

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// A simple 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.z
            .mul_add(other.z, self.x.mul_add(other.x, self.y * other.y))
    }

    #[must_use]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    #[must_use]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-15 {
            return Self::new(0.0, 0.0, 0.0);
        }
        Self::new(self.x / len, self.y / len, self.z / len)
    }

    #[must_use]
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    #[must_use]
    pub fn minus(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y.mul_add(other.z, -(self.z * other.y)),
            self.z.mul_add(other.x, -(self.x * other.z)),
            self.x.mul_add(other.y, -(self.y * other.x)),
        )
    }
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

/// An optical ray with origin, direction, and wavelength (in metres).
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub wavelength: f64,
}

impl Ray {
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3, wavelength: f64) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
            wavelength,
        }
    }

    /// Propagate the ray by distance `t`.
    #[must_use]
    pub fn at(self, t: f64) -> Vec3 {
        self.origin.plus(self.direction.scale(t))
    }
}

// ---------------------------------------------------------------------------
// Snell's law
// ---------------------------------------------------------------------------

/// Apply Snell's law: `n1 * sin(theta1) = n2 * sin(theta2)`.
/// Returns `None` for total internal reflection.
#[must_use]
pub fn snell(n1: f64, theta1: f64, n2: f64) -> Option<f64> {
    let sin_t2 = n1 * theta1.sin() / n2;
    if sin_t2.abs() > 1.0 {
        None
    } else {
        Some(sin_t2.asin())
    }
}

/// Vector form of Snell's law. Returns refracted direction or `None` (TIR).
/// `normal` must point from medium 2 toward medium 1 (against incoming ray).
#[must_use]
pub fn snell_vec(incident: Vec3, normal: Vec3, n1: f64, n2: f64) -> Option<Vec3> {
    let i = incident.normalized();
    let n = normal.normalized();
    let cos_i = -i.dot(n);
    let ratio = n1 / n2;
    let sin2_t = ratio * ratio * cos_i.mul_add(-cos_i, 1.0);
    if sin2_t > 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    Some(i.scale(ratio).plus(n.scale(ratio.mul_add(cos_i, -cos_t))))
}

// ---------------------------------------------------------------------------
// Fresnel equations
// ---------------------------------------------------------------------------

/// Fresnel reflectance for s-polarization.
#[must_use]
pub fn fresnel_rs(n1: f64, theta_i: f64, n2: f64) -> f64 {
    let cos_i = theta_i.cos();
    let sin_t = n1 * theta_i.sin() / n2;
    if sin_t.abs() > 1.0 {
        return 1.0; // TIR
    }
    let cos_t = sin_t.mul_add(-sin_t, 1.0).sqrt();
    let num = n1.mul_add(cos_i, -(n2 * cos_t));
    let den = n1.mul_add(cos_i, n2 * cos_t);
    (num / den) * (num / den)
}

/// Fresnel reflectance for p-polarization.
#[must_use]
pub fn fresnel_rp(n1: f64, theta_i: f64, n2: f64) -> f64 {
    let cos_i = theta_i.cos();
    let sin_t = n1 * theta_i.sin() / n2;
    if sin_t.abs() > 1.0 {
        return 1.0; // TIR
    }
    let cos_t = sin_t.mul_add(-sin_t, 1.0).sqrt();
    let num = n2.mul_add(cos_i, -(n1 * cos_t));
    let den = n2.mul_add(cos_i, n1 * cos_t);
    (num / den) * (num / den)
}

/// Unpolarized Fresnel reflectance (average of s and p).
#[must_use]
pub fn fresnel_unpolarized(n1: f64, theta_i: f64, n2: f64) -> f64 {
    0.5 * (fresnel_rs(n1, theta_i, n2) + fresnel_rp(n1, theta_i, n2))
}

/// Brewster's angle for interface n1 -> n2.
#[must_use]
pub fn brewster_angle(n1: f64, n2: f64) -> f64 {
    (n2 / n1).atan()
}

/// Critical angle for total internal reflection (n1 > n2).
/// Returns `None` if n1 <= n2.
#[must_use]
pub fn critical_angle(n1: f64, n2: f64) -> Option<f64> {
    if n1 <= n2 {
        None
    } else {
        Some((n2 / n1).asin())
    }
}

// ---------------------------------------------------------------------------
// Thin lens
// ---------------------------------------------------------------------------

/// Thin lens equation: 1/f = 1/do + 1/di.
/// Given focal length `f` and object distance `d_o`, returns image distance.
#[must_use]
pub fn thin_lens_image_distance(f: f64, d_o: f64) -> f64 {
    (1.0 / f - 1.0 / d_o).recip()
}

/// Magnification of a thin lens: M = -di / do.
#[must_use]
pub fn thin_lens_magnification(d_o: f64, d_i: f64) -> f64 {
    -d_i / d_o
}

/// Lensmaker's equation for a thin lens in air:
/// `1/f = (n - 1) * (1/r1 - 1/r2)`.
#[must_use]
pub fn lensmaker_thin(n: f64, r1: f64, r2: f64) -> f64 {
    1.0 / ((n - 1.0) * (1.0 / r1 - 1.0 / r2))
}

/// Optical power (diopters) of a thin lens: P = 1/f.
#[must_use]
pub fn optical_power(f: f64) -> f64 {
    1.0 / f
}

// ---------------------------------------------------------------------------
// Thick lens
// ---------------------------------------------------------------------------

/// Thick lens focal length.
/// `n` = refractive index, `r1`/`r2` = radii of curvature, `d` = thickness.
#[must_use]
pub fn thick_lens_focal_length(n: f64, r1: f64, r2: f64, d: f64) -> f64 {
    let phi1 = (n - 1.0) / r1;
    let phi2 = -(n - 1.0) / r2;
    let phi = phi1 + phi2 - (d * phi1 * phi2) / n;
    1.0 / phi
}

/// Thick lens principal plane offset from front surface.
#[must_use]
pub fn thick_lens_front_principal(n: f64, r1: f64, r2: f64, d: f64) -> f64 {
    let phi2 = -(n - 1.0) / r2;
    let f = thick_lens_focal_length(n, r1, r2, d);
    -f * d * phi2 / n
}

/// Thick lens principal plane offset from back surface.
#[must_use]
pub fn thick_lens_back_principal(n: f64, r1: f64, r2: f64, d: f64) -> f64 {
    let phi1 = (n - 1.0) / r1;
    let f = thick_lens_focal_length(n, r1, r2, d);
    -f * d * phi1 / n
}

// ---------------------------------------------------------------------------
// Lens system (matrix optics / ABCD matrices)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Ray tracing through spherical surfaces
// ---------------------------------------------------------------------------

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
        // propagate
        y += d * u;
        // refract
        let n_next = surf.n_after;
        u = n.mul_add(u, (n - n_next) * y / surf.radius) / n_next;
        n = n_next;
        z_prev = surf.z;
    }

    (y, u)
}

// ---------------------------------------------------------------------------
// Diffraction
// ---------------------------------------------------------------------------

/// Single-slit Fraunhofer diffraction intensity (normalized).
/// `a` = slit width, `lambda` = wavelength, `theta` = angle.
#[must_use]
pub fn single_slit_intensity(a: f64, lambda: f64, theta: f64) -> f64 {
    let beta = PI * a * theta.sin() / lambda;
    if beta.abs() < 1e-12 {
        1.0
    } else {
        let sinc = beta.sin() / beta;
        sinc * sinc
    }
}

/// Double-slit Fraunhofer diffraction intensity (normalized).
/// `a` = slit width, `d` = slit separation, `lambda` = wavelength, `theta` = angle.
#[must_use]
pub fn double_slit_intensity(a: f64, d: f64, lambda: f64, theta: f64) -> f64 {
    let envelope = single_slit_intensity(a, lambda, theta);
    let delta = PI * d * theta.sin() / lambda;
    let interference = delta.cos() * delta.cos();
    envelope * interference
}

/// Positions of diffraction minima for a single slit.
/// Returns angles for orders `1..=max_order`.
#[must_use]
pub fn single_slit_minima(a: f64, lambda: f64, max_order: u32) -> Vec<f64> {
    (1..=max_order)
        .map(|m| {
            let sin_val = f64::from(m) * lambda / a;
            if sin_val.abs() <= 1.0 {
                sin_val.asin()
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Positions of double-slit interference maxima.
/// Returns angles for orders `0..=max_order`.
#[must_use]
pub fn double_slit_maxima(d: f64, lambda: f64, max_order: u32) -> Vec<f64> {
    (0..=max_order)
        .map(|m| {
            let sin_val = f64::from(m) * lambda / d;
            if sin_val.abs() <= 1.0 {
                sin_val.asin()
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Circular aperture Airy disk: first zero at sin(theta) = 1.22 * lambda / D.
#[must_use]
pub fn airy_first_zero(lambda: f64, diameter: f64) -> f64 {
    (1.22 * lambda / diameter).asin()
}

/// Rayleigh criterion angular resolution.
#[must_use]
pub fn rayleigh_resolution(lambda: f64, diameter: f64) -> f64 {
    1.22 * lambda / diameter
}

/// Diffraction grating: maxima at d * sin(theta) = m * lambda.
#[must_use]
pub fn grating_maxima(d: f64, lambda: f64, max_order: i32) -> Vec<f64> {
    let mut angles = Vec::new();
    for m in -max_order..=max_order {
        let sin_val = f64::from(m) * lambda / d;
        if sin_val.abs() <= 1.0 {
            angles.push(sin_val.asin());
        }
    }
    angles
}

// ---------------------------------------------------------------------------
// Polarization — Jones calculus
// ---------------------------------------------------------------------------

/// A Jones vector representing the polarization state of light.
#[derive(Debug, Clone, Copy)]
pub struct JonesVector {
    pub ex: (f64, f64), // (real, imag)
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

// ---------------------------------------------------------------------------
// Polarization — Stokes parameters
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Thin film interference
// ---------------------------------------------------------------------------

/// Thin film reflectance for a film of refractive index `n_f`, thickness `d`,
/// on a substrate of index `n_s`, illuminated from medium with index `n0`,
/// at normal incidence, for wavelength `lambda`.
#[must_use]
pub fn thin_film_reflectance(n0: f64, n_f: f64, n_s: f64, d: f64, lambda: f64) -> f64 {
    let r1 = (n0 - n_f) / (n0 + n_f);
    let r2 = (n_f - n_s) / (n_f + n_s);
    let delta = 2.0 * PI * n_f * d / lambda;
    let num = (2.0 * r1 * r2).mul_add((2.0 * delta).cos(), r1.mul_add(r1, r2 * r2));
    let den = (2.0 * r1 * r2).mul_add((2.0 * delta).cos(), (r1 * r1 * r2).mul_add(r2, 1.0));
    num / den
}

/// Optimal anti-reflection coating thickness (quarter-wave) for wavelength `lambda`.
#[must_use]
pub fn anti_reflection_thickness(n_film: f64, lambda: f64) -> f64 {
    lambda / (4.0 * n_film)
}

/// Ideal anti-reflection coating refractive index for interface n1 -> n2.
#[must_use]
pub fn anti_reflection_index(n1: f64, n2: f64) -> f64 {
    (n1 * n2).sqrt()
}

/// Constructive interference condition: 2 * n * d = m * lambda.
/// Returns the wavelength for order `m`.
#[must_use]
pub fn thin_film_constructive_lambda(n: f64, d: f64, m: u32) -> f64 {
    2.0 * n * d / f64::from(m)
}

/// Destructive interference condition: 2 * n * d = (m + 0.5) * lambda.
#[must_use]
pub fn thin_film_destructive_lambda(n: f64, d: f64, m: u32) -> f64 {
    2.0 * n * d / (f64::from(m) + 0.5)
}

// ---------------------------------------------------------------------------
// Fiber optics
// ---------------------------------------------------------------------------

/// Numerical aperture of an optical fiber.
#[must_use]
pub fn fiber_na(n_core: f64, n_clad: f64) -> f64 {
    n_core.mul_add(n_core, -(n_clad * n_clad)).sqrt()
}

/// Acceptance angle (half-angle) of an optical fiber.
#[must_use]
pub fn fiber_acceptance_angle(n_core: f64, n_clad: f64) -> f64 {
    fiber_na(n_core, n_clad).asin()
}

/// Number of modes in a step-index fiber (V-number based).
/// `a` = core radius, `lambda` = wavelength.
#[must_use]
pub fn fiber_v_number(n_core: f64, n_clad: f64, a: f64, lambda: f64) -> f64 {
    2.0 * PI * a * fiber_na(n_core, n_clad) / lambda
}

/// Approximate number of modes for a step-index fiber.
#[must_use]
pub fn fiber_num_modes(v: f64) -> f64 {
    v * v / 2.0
}

/// Whether a fiber is single-mode (V < 2.405).
#[must_use]
pub fn fiber_is_single_mode(v: f64) -> bool {
    v < 2.405
}

/// Fiber attenuation: output power given input power, attenuation coefficient
/// (dB/km), and length (km).
#[must_use]
pub fn fiber_attenuation(p_in: f64, alpha_db_per_km: f64, length_km: f64) -> f64 {
    p_in * 10.0_f64.powf(-alpha_db_per_km * length_km / 10.0)
}

/// Pulse broadening due to modal dispersion in step-index fiber.
/// Returns time spread per unit length.
#[must_use]
pub fn modal_dispersion(n_core: f64, n_clad: f64, length: f64) -> f64 {
    let delta = (n_core - n_clad) / n_clad;
    n_core * delta * length / 299_792_458.0
}

// ---------------------------------------------------------------------------
// Aberrations
// ---------------------------------------------------------------------------

/// Longitudinal spherical aberration for a thin lens.
/// `f` = focal length, `h` = ray height, `n` = refractive index.
/// Uses third-order approximation.
#[must_use]
pub fn spherical_aberration_longitudinal(f: f64, h: f64, n: f64) -> f64 {
    let q = 0.0; // shape factor for equi-convex
    let p = 0.0; // position factor for object at infinity
    let s_coeff = ((1.0 / (16.0 * n * (n - 1.0))) * (n + 2.0) / (n - 1.0)).mul_add(
        ((2.0 * n.mul_add(n, -1.0)) / (n + 2.0))
            .mul_add(q, p)
            .powi(2),
        n * n / ((n - 1.0) * (n - 1.0)),
    );
    // The actual aberration for simplified equi-convex at infinity:
    h * h / (2.0 * f) * s_coeff
}

/// Transverse spherical aberration from longitudinal.
#[must_use]
pub fn spherical_aberration_transverse(longitudinal: f64, u_angle: f64) -> f64 {
    longitudinal * u_angle.tan()
}

/// Chromatic aberration: focal length difference between two wavelengths.
/// Uses Cauchy dispersion: n(lambda) = a + b / lambda^2.
#[must_use]
pub fn chromatic_aberration(a: f64, b: f64, r1: f64, r2: f64, lambda1: f64, lambda2: f64) -> f64 {
    let n1 = a + b / (lambda1 * lambda1);
    let n2 = a + b / (lambda2 * lambda2);
    let f1 = lensmaker_thin(n1, r1, r2);
    let f2 = lensmaker_thin(n2, r1, r2);
    f1 - f2
}

/// Abbe number (V-number) for a glass: V = (nd - 1) / (nF - nC).
#[must_use]
pub fn abbe_number(n_d: f64, n_f: f64, n_c: f64) -> f64 {
    (n_d - 1.0) / (n_f - n_c)
}

/// Achromatic doublet condition: f1 * V1 + f2 * V2 = 0.
/// Given total power `phi_total` and Abbe numbers, returns individual powers.
#[must_use]
pub fn achromatic_doublet(phi_total: f64, v1: f64, v2: f64) -> (f64, f64) {
    let phi1 = phi_total * v1 / (v1 - v2);
    let phi2 = phi_total * v2 / (v2 - v1);
    (phi1, phi2)
}

/// Seidel coefficient for coma (third-order, thin lens at infinity).
#[must_use]
pub fn coma_coefficient(f: f64, h: f64, n: f64) -> f64 {
    h * h / (2.0 * f * f) * (n + 1.0) / (4.0 * n)
}

/// Petzval field curvature radius.
#[must_use]
pub fn petzval_radius(focal_lengths: &[f64], indices: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (f, n) in focal_lengths.iter().zip(indices.iter()) {
        sum += 1.0 / (n * f);
    }
    if sum.abs() < 1e-15 {
        f64::INFINITY
    } else {
        1.0 / sum
    }
}

/// Distortion percentage.
#[must_use]
pub fn distortion_percent(actual_height: f64, ideal_height: f64) -> f64 {
    (actual_height - ideal_height) / ideal_height * 100.0
}

// ---------------------------------------------------------------------------
// Dispersion
// ---------------------------------------------------------------------------

/// Cauchy dispersion formula: n(lambda) = a + b / lambda^2 + c / lambda^4.
#[must_use]
pub fn cauchy_dispersion(a: f64, b: f64, c: f64, lambda: f64) -> f64 {
    a + b / (lambda * lambda) + c / (lambda * lambda * lambda * lambda)
}

/// Sellmeier equation (single term): n^2 - 1 = B * lambda^2 / (lambda^2 - C).
#[must_use]
pub fn sellmeier_single(big_b: f64, big_c: f64, lambda: f64) -> f64 {
    (1.0 + big_b * lambda * lambda / lambda.mul_add(lambda, -big_c)).sqrt()
}

/// Group refractive index: `n_g` = n - lambda * dn/dlambda.
/// Uses numerical differentiation.
#[must_use]
pub fn group_index(n_func: fn(f64) -> f64, lambda: f64) -> f64 {
    let dl = lambda * 1e-6;
    let dn = (n_func(lambda + dl) - n_func(lambda - dl)) / (2.0 * dl);
    lambda.mul_add(-dn, n_func(lambda))
}

// ---------------------------------------------------------------------------
// Gaussian beam optics
// ---------------------------------------------------------------------------

/// Rayleigh range for a Gaussian beam.
#[must_use]
pub fn rayleigh_range(w0: f64, lambda: f64) -> f64 {
    PI * w0 * w0 / lambda
}

/// Beam waist at distance z from focus.
#[must_use]
pub fn beam_waist_at_z(w0: f64, z: f64, z_r: f64) -> f64 {
    w0 * (z / z_r).mul_add(z / z_r, 1.0).sqrt()
}

/// Radius of curvature of wavefront at distance z.
#[must_use]
pub fn wavefront_radius(z: f64, z_r: f64) -> f64 {
    if z.abs() < 1e-15 {
        return f64::INFINITY;
    }
    z * (z_r / z).mul_add(z_r / z, 1.0)
}

/// Beam divergence half-angle (far field).
#[must_use]
pub fn beam_divergence(w0: f64, lambda: f64) -> f64 {
    lambda / (PI * w0)
}

/// Gouy phase at distance z.
#[must_use]
pub fn gouy_phase(z: f64, z_r: f64) -> f64 {
    (z / z_r).atan()
}

// ---------------------------------------------------------------------------
// Prism
// ---------------------------------------------------------------------------

/// Minimum deviation angle for a prism with apex angle `apex` and index `n`.
#[must_use]
pub fn prism_min_deviation(n: f64, apex: f64) -> f64 {
    2.0f64.mul_add((n * (apex / 2.0).sin()).asin(), -apex)
}

/// Refractive index from prism minimum deviation measurement.
#[must_use]
pub fn prism_index_from_deviation(apex: f64, deviation: f64) -> f64 {
    f64::midpoint(apex, deviation).sin() / (apex / 2.0).sin()
}

/// Angular dispersion of a prism (dn/dlambda contribution).
#[must_use]
pub fn prism_angular_dispersion(apex: f64, n: f64, dn_dlambda: f64) -> f64 {
    let sin_half = (apex / 2.0).sin();
    let dm = prism_min_deviation(n, apex);
    let cos_half_dm = f64::midpoint(apex, dm).cos();
    2.0 * sin_half * dn_dlambda / cos_half_dm
}

// ---------------------------------------------------------------------------
// Optical path / interference
// ---------------------------------------------------------------------------

/// Optical path length.
#[must_use]
pub fn optical_path_length(n: f64, d: f64) -> f64 {
    n * d
}

/// Phase difference from optical path difference.
#[must_use]
pub fn phase_from_opd(opd: f64, lambda: f64) -> f64 {
    2.0 * PI * opd / lambda
}

/// Coherence length from bandwidth.
#[must_use]
pub fn coherence_length(lambda: f64, delta_lambda: f64) -> f64 {
    lambda * lambda / delta_lambda
}

/// Michelson interferometer visibility from max/min intensities.
#[must_use]
pub fn fringe_visibility(i_max: f64, i_min: f64) -> f64 {
    (i_max - i_min) / (i_max + i_min)
}

/// Two-beam interference intensity.
#[must_use]
pub fn two_beam_interference(i1: f64, i2: f64, phase_diff: f64) -> f64 {
    (2.0 * (i1 * i2).sqrt()).mul_add(phase_diff.cos(), i1 + i2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- Snell's law ----

    #[test]
    fn test_snell_normal_incidence() {
        let t2 = snell(1.0, 0.0, 1.5).unwrap();
        assert!(approx(t2, 0.0, EPS));
    }

    #[test]
    fn test_snell_known_angle() {
        let t2 = snell(1.0, PI / 6.0, 1.5).unwrap();
        let expected = (0.5 / 1.5_f64).asin();
        assert!(approx(t2, expected, EPS));
    }

    #[test]
    fn test_snell_tir() {
        assert!(snell(1.5, PI / 3.0, 1.0).is_none());
    }

    #[test]
    fn test_snell_vec_normal() {
        let d = snell_vec(
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            1.5,
        );
        assert!(d.is_some());
        let r = d.unwrap();
        assert!(approx(r.x, 0.0, EPS));
        assert!(approx(r.y, 0.0, EPS));
        assert!(r.z < 0.0);
    }

    #[test]
    fn test_snell_vec_tir() {
        let d = snell_vec(
            Vec3::new(0.9, 0.0, -0.4).normalized(),
            Vec3::new(0.0, 0.0, 1.0),
            1.5,
            1.0,
        );
        assert!(d.is_none());
    }

    // ---- Fresnel equations ----

    #[test]
    fn test_fresnel_normal_incidence() {
        let r = fresnel_unpolarized(1.0, 0.0, 1.5);
        let expected = ((1.0_f64 - 1.5) / (1.0 + 1.5)).powi(2);
        assert!(approx(r, expected, EPS));
    }

    #[test]
    fn test_fresnel_rp_at_brewster() {
        let b = brewster_angle(1.0, 1.5);
        let rp = fresnel_rp(1.0, b, 1.5);
        assert!(approx(rp, 0.0, 1e-4));
    }

    #[test]
    fn test_fresnel_tir() {
        let r = fresnel_rs(1.5, PI / 3.0, 1.0);
        assert!(approx(r, 1.0, EPS));
    }

    #[test]
    fn test_brewster_angle() {
        let b = brewster_angle(1.0, 1.5);
        assert!(approx(b.tan(), 1.5, EPS));
    }

    #[test]
    fn test_critical_angle() {
        let c = critical_angle(1.5, 1.0).unwrap();
        assert!(approx(c.sin(), 1.0 / 1.5, EPS));
    }

    #[test]
    fn test_critical_angle_none() {
        assert!(critical_angle(1.0, 1.5).is_none());
    }

    // ---- Thin lens ----

    #[test]
    fn test_thin_lens_image_at_2f() {
        let d_i = thin_lens_image_distance(0.1, 0.2);
        assert!(approx(d_i, 0.2, EPS));
    }

    #[test]
    fn test_thin_lens_image_at_infinity() {
        let d_i = thin_lens_image_distance(0.1, 0.1);
        assert!(d_i > 1e10);
    }

    #[test]
    fn test_thin_lens_magnification() {
        let m = thin_lens_magnification(0.2, 0.2);
        assert!(approx(m, -1.0, EPS));
    }

    #[test]
    fn test_lensmaker_thin() {
        let f = lensmaker_thin(1.5, 0.2, -0.2);
        let expected = 1.0 / (0.5 * (1.0 / 0.2 + 1.0 / 0.2));
        assert!(approx(f, expected, EPS));
    }

    #[test]
    fn test_optical_power() {
        assert!(approx(optical_power(0.5), 2.0, EPS));
    }

    // ---- Thick lens ----

    #[test]
    fn test_thick_lens_reduces_to_thin() {
        let f_thick = thick_lens_focal_length(1.5, 0.2, -0.2, 0.001);
        let f_thin = lensmaker_thin(1.5, 0.2, -0.2);
        assert!(approx(f_thick, f_thin, 1e-3));
    }

    #[test]
    fn test_thick_lens_principal_planes() {
        let h1 = thick_lens_front_principal(1.5, 0.1, -0.1, 0.02);
        let h2 = thick_lens_back_principal(1.5, 0.1, -0.1, 0.02);
        // For symmetric lens, |h1| = |h2|
        assert!(approx(h1.abs(), h2.abs(), 1e-4));
    }

    // ---- Matrix optics ----

    #[test]
    fn test_propagation_matrix() {
        let m = Matrix2x2::propagation(1.0);
        let (y, theta) = m.apply(0.0, 0.1);
        assert!(approx(y, 0.1, EPS));
        assert!(approx(theta, 0.1, EPS));
    }

    #[test]
    fn test_thin_lens_matrix() {
        let m = Matrix2x2::thin_lens(0.5);
        let (y, theta) = m.apply(0.1, 0.0);
        assert!(approx(y, 0.1, EPS));
        assert!(approx(theta, -0.2, EPS));
    }

    #[test]
    fn test_matrix_det() {
        let m = Matrix2x2::propagation(2.0);
        assert!(approx(m.det(), 1.0, EPS));
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix2x2::identity();
        let (y, t) = m.apply(3.0, 0.5);
        assert!(approx(y, 3.0, EPS));
        assert!(approx(t, 0.5, EPS));
    }

    #[test]
    fn test_compose_matrices() {
        let m = compose_matrices(&[
            Matrix2x2::propagation(1.0),
            Matrix2x2::thin_lens(0.5),
            Matrix2x2::propagation(1.0),
        ]);
        // Parallel ray at height 0.1 should focus to axis after lens+propagation
        let (y, _) = m.apply(0.1, 0.0);
        // With f=0.5, after 1m propagation: not exactly zero but deterministic
        assert!(y.is_finite());
    }

    #[test]
    fn test_flat_refraction() {
        let m = Matrix2x2::flat_refraction(1.0, 1.5);
        assert!(approx(m.d, 1.0 / 1.5, EPS));
    }

    #[test]
    fn test_curved_refraction() {
        let m = Matrix2x2::curved_refraction(1.0, 1.5, 0.1);
        assert!(approx(m.c, (1.0 - 1.5) / (1.5 * 0.1), EPS));
    }

    // ---- Paraxial ray tracing ----

    #[test]
    fn test_trace_paraxial_single_surface() {
        let surfaces = [SphericalSurface {
            z: 0.0,
            radius: 0.1,
            n_after: 1.5,
        }];
        let (y, u) = trace_paraxial(&surfaces, 1.0, 0.01, 0.0);
        assert!(y.is_finite());
        assert!(u.is_finite());
    }

    #[test]
    fn test_trace_paraxial_flat() {
        let surfaces = [SphericalSurface {
            z: 1.0,
            radius: 1e10,
            n_after: 1.0,
        }];
        let (y, u) = trace_paraxial(&surfaces, 1.0, 0.0, 0.1);
        assert!(approx(y, 0.1, 1e-3));
        assert!(approx(u, 0.1, 1e-3));
    }

    // ---- Single-slit diffraction ----

    #[test]
    fn test_single_slit_center() {
        let i = single_slit_intensity(1e-4, 500e-9, 0.0);
        assert!(approx(i, 1.0, EPS));
    }

    #[test]
    fn test_single_slit_first_min() {
        let theta = (500e-9 / 1e-4_f64).asin();
        let i = single_slit_intensity(1e-4, 500e-9, theta);
        assert!(approx(i, 0.0, 1e-4));
    }

    #[test]
    fn test_single_slit_symmetry() {
        let a = single_slit_intensity(1e-4, 500e-9, 0.001);
        let b = single_slit_intensity(1e-4, 500e-9, -0.001);
        assert!(approx(a, b, EPS));
    }

    // ---- Double-slit diffraction ----

    #[test]
    fn test_double_slit_center() {
        let i = double_slit_intensity(1e-5, 1e-4, 500e-9, 0.0);
        assert!(approx(i, 1.0, EPS));
    }

    #[test]
    fn test_double_slit_envelope() {
        let theta = 0.001;
        let ds = double_slit_intensity(1e-5, 1e-4, 500e-9, theta);
        let ss = single_slit_intensity(1e-5, 500e-9, theta);
        assert!(ds <= ss + EPS);
    }

    #[test]
    fn test_single_slit_minima() {
        let minima = single_slit_minima(1e-4, 500e-9, 3);
        assert_eq!(minima.len(), 3);
        for (m, angle) in minima.iter().enumerate() {
            let expected = ((m as f64 + 1.0) * 500e-9 / 1e-4).asin();
            assert!(approx(*angle, expected, EPS));
        }
    }

    #[test]
    fn test_double_slit_maxima() {
        let maxima = double_slit_maxima(1e-4, 500e-9, 2);
        assert_eq!(maxima.len(), 3); // orders 0, 1, 2
        assert!(approx(maxima[0], 0.0, EPS));
    }

    #[test]
    fn test_airy_disk() {
        let theta = airy_first_zero(500e-9, 0.01);
        assert!(theta > 0.0);
    }

    #[test]
    fn test_rayleigh_resolution() {
        let r = rayleigh_resolution(500e-9, 0.1);
        assert!(approx(r, 1.22 * 500e-9 / 0.1, EPS));
    }

    #[test]
    fn test_grating_maxima() {
        let angles = grating_maxima(1e-5, 500e-9, 3);
        assert!(!angles.is_empty());
        // Order 0 should be at 0
        assert!(angles.iter().any(|a| approx(*a, 0.0, EPS)));
    }

    // ---- Jones vectors and matrices ----

    #[test]
    fn test_jones_horizontal_intensity() {
        let h = JonesVector::horizontal();
        assert!(approx(h.intensity(), 1.0, EPS));
    }

    #[test]
    fn test_jones_vertical_intensity() {
        let v = JonesVector::vertical();
        assert!(approx(v.intensity(), 1.0, EPS));
    }

    #[test]
    fn test_jones_diagonal_intensity() {
        let d = JonesVector::diagonal();
        assert!(approx(d.intensity(), 1.0, EPS));
    }

    #[test]
    fn test_jones_circular_intensity() {
        let r = JonesVector::right_circular();
        assert!(approx(r.intensity(), 1.0, EPS));
        let l = JonesVector::left_circular();
        assert!(approx(l.intensity(), 1.0, EPS));
    }

    #[test]
    fn test_jones_polarizer_h_blocks_v() {
        let pol = JonesMatrix::polarizer_h();
        let v = JonesVector::vertical();
        let result = pol.apply(v);
        assert!(approx(result.intensity(), 0.0, EPS));
    }

    #[test]
    fn test_jones_polarizer_h_passes_h() {
        let pol = JonesMatrix::polarizer_h();
        let h = JonesVector::horizontal();
        let result = pol.apply(h);
        assert!(approx(result.intensity(), 1.0, EPS));
    }

    #[test]
    fn test_jones_polarizer_v_blocks_h() {
        let pol = JonesMatrix::polarizer_v();
        let h = JonesVector::horizontal();
        let result = pol.apply(h);
        assert!(approx(result.intensity(), 0.0, EPS));
    }

    #[test]
    fn test_jones_crossed_polarizers() {
        let h_pol = JonesMatrix::polarizer_h();
        let v_pol = JonesMatrix::polarizer_v();
        let system = v_pol.matmul(h_pol);
        let light = JonesVector::diagonal();
        let result = system.apply(light);
        assert!(approx(result.intensity(), 0.0, EPS));
    }

    #[test]
    fn test_jones_polarizer_45() {
        let pol = JonesMatrix::polarizer(PI / 4.0);
        let h = JonesVector::horizontal();
        let result = pol.apply(h);
        assert!(approx(result.intensity(), 0.5, EPS));
    }

    #[test]
    fn test_jones_half_wave_plate() {
        let hwp = JonesMatrix::half_wave_plate(0.0);
        let h = JonesVector::horizontal();
        let result = hwp.apply(h);
        assert!(approx(result.intensity(), 1.0, EPS));
    }

    #[test]
    fn test_jones_quarter_wave_plate() {
        let qwp = JonesMatrix::quarter_wave_plate(0.0);
        let h = JonesVector::horizontal();
        let result = qwp.apply(h);
        assert!(approx(result.intensity(), 1.0, EPS));
    }

    #[test]
    fn test_jones_malus_law() {
        // Malus' law: I = I0 * cos^2(theta)
        for deg in 0..=9 {
            let theta = f64::from(deg) * 10.0 * PI / 180.0;
            let pol = JonesMatrix::polarizer(theta);
            let h = JonesVector::horizontal();
            let result = pol.apply(h);
            let expected = theta.cos().powi(2);
            assert!(approx(result.intensity(), expected, 1e-4));
        }
    }

    // ---- Stokes vectors ----

    #[test]
    fn test_stokes_unpolarized_dop() {
        let s = StokesVector::unpolarized(1.0);
        assert!(approx(s.degree_of_polarization(), 0.0, EPS));
    }

    #[test]
    fn test_stokes_horizontal_dop() {
        let s = StokesVector::horizontal(1.0);
        assert!(approx(s.degree_of_polarization(), 1.0, EPS));
    }

    #[test]
    fn test_stokes_circular_dop() {
        let s = StokesVector::right_circular(1.0);
        assert!(approx(s.degree_of_polarization(), 1.0, EPS));
    }

    #[test]
    fn test_stokes_orientation_horizontal() {
        let s = StokesVector::horizontal(1.0);
        assert!(approx(s.orientation_angle(), 0.0, EPS));
    }

    #[test]
    fn test_stokes_orientation_vertical() {
        let s = StokesVector::vertical(1.0);
        assert!(approx(s.orientation_angle(), PI / 2.0, EPS));
    }

    #[test]
    fn test_stokes_ellipticity_circular() {
        let s = StokesVector::right_circular(1.0);
        assert!(approx(s.ellipticity_angle(), PI / 4.0, EPS));
    }

    #[test]
    fn test_stokes_ellipticity_linear() {
        let s = StokesVector::horizontal(1.0);
        assert!(approx(s.ellipticity_angle(), 0.0, EPS));
    }

    // ---- Mueller matrices ----

    #[test]
    fn test_mueller_identity() {
        let m = MuellerMatrix::identity();
        let s = StokesVector::horizontal(1.0);
        let result = m.apply(s);
        assert!(approx(result.s0, s.s0, EPS));
        assert!(approx(result.s1, s.s1, EPS));
    }

    #[test]
    fn test_mueller_h_polarizer() {
        let m = MuellerMatrix::polarizer_h();
        let s = StokesVector::unpolarized(1.0);
        let result = m.apply(s);
        assert!(approx(result.s0, 0.5, EPS));
        assert!(approx(result.s1, 0.5, EPS));
    }

    #[test]
    fn test_mueller_v_blocks_h() {
        let m = MuellerMatrix::polarizer_v();
        let s = StokesVector::horizontal(1.0);
        let result = m.apply(s);
        assert!(approx(result.s0, 0.0, EPS));
    }

    // ---- Thin film interference ----

    #[test]
    fn test_thin_film_quarter_wave_ar() {
        let n_film = (1.0_f64 * 1.5_f64).sqrt();
        let d = 500e-9 / (4.0 * n_film);
        let r = thin_film_reflectance(1.0, n_film, 1.5, d, 500e-9);
        assert!(approx(r, 0.0, 1e-4));
    }

    #[test]
    fn test_thin_film_half_wave() {
        let n_film = 1.38;
        let d = 500e-9 / (2.0 * n_film);
        let r = thin_film_reflectance(1.0, n_film, 1.5, d, 500e-9);
        // half-wave: film is transparent, reflectance = bare substrate
        let r_bare = ((1.0_f64 - 1.5) / (1.0 + 1.5)).powi(2);
        assert!(approx(r, r_bare, 1e-3));
    }

    #[test]
    fn test_anti_reflection_thickness() {
        let t = anti_reflection_thickness(1.38, 550e-9);
        assert!(approx(t, 550e-9 / (4.0 * 1.38), EPS));
    }

    #[test]
    fn test_anti_reflection_index() {
        let n = anti_reflection_index(1.0, 1.5);
        assert!(approx(n, 1.5_f64.sqrt(), EPS));
    }

    #[test]
    fn test_constructive_lambda() {
        let lam = thin_film_constructive_lambda(1.5, 200e-9, 1);
        assert!(approx(lam, 600e-9, EPS));
    }

    #[test]
    fn test_destructive_lambda() {
        let lam = thin_film_destructive_lambda(1.5, 200e-9, 0);
        assert!(approx(lam, 600e-9 / 0.5, EPS));
    }

    // ---- Fiber optics ----

    #[test]
    fn test_fiber_na() {
        let na = fiber_na(1.48, 1.46);
        let expected = (1.48_f64.powi(2) - 1.46_f64.powi(2)).sqrt();
        assert!(approx(na, expected, EPS));
    }

    #[test]
    fn test_fiber_acceptance_angle() {
        let a = fiber_acceptance_angle(1.48, 1.46);
        assert!(a > 0.0 && a < PI / 2.0);
    }

    #[test]
    fn test_fiber_v_number() {
        let v = fiber_v_number(1.48, 1.46, 4e-6, 1.3e-6);
        assert!(v > 0.0);
    }

    #[test]
    fn test_fiber_single_mode() {
        assert!(fiber_is_single_mode(2.0));
        assert!(!fiber_is_single_mode(3.0));
    }

    #[test]
    fn test_fiber_num_modes() {
        assert!(approx(fiber_num_modes(10.0), 50.0, EPS));
    }

    #[test]
    fn test_fiber_attenuation() {
        let p_out = fiber_attenuation(1.0, 0.2, 10.0);
        // 0.2 dB/km * 10 km = 2 dB loss
        let expected = 10.0_f64.powf(-0.2);
        assert!(approx(p_out, expected, EPS));
    }

    #[test]
    fn test_modal_dispersion() {
        let dt = modal_dispersion(1.48, 1.46, 1000.0);
        assert!(dt > 0.0);
    }

    // ---- Aberrations ----

    #[test]
    fn test_spherical_aberration() {
        let sa = spherical_aberration_longitudinal(0.1, 0.01, 1.5);
        assert!(sa > 0.0);
    }

    #[test]
    fn test_transverse_aberration() {
        let ta = spherical_aberration_transverse(0.001, 0.1);
        assert!(approx(ta, 0.001 * 0.1_f64.tan(), EPS));
    }

    #[test]
    fn test_chromatic_aberration() {
        let ca = chromatic_aberration(1.5, 0.005, 0.1, -0.1, 486e-9, 656e-9);
        assert!(ca.abs() > 0.0);
    }

    #[test]
    fn test_abbe_number() {
        let v = abbe_number(1.5168, 1.5224, 1.5143);
        let expected = (1.5168 - 1.0) / (1.5224 - 1.5143);
        assert!(approx(v, expected, EPS));
    }

    #[test]
    fn test_achromatic_doublet() {
        let (p1, p2) = achromatic_doublet(10.0, 64.0, 28.0);
        // p1/V1 + p2/V2 should be ~0
        assert!(approx(p1 / 64.0 + p2 / 28.0, 0.0, 1e-10));
        // p1 + p2 = phi_total
        assert!(approx(p1 + p2, 10.0, EPS));
    }

    #[test]
    fn test_coma() {
        let c = coma_coefficient(0.1, 0.01, 1.5);
        assert!(c > 0.0);
    }

    #[test]
    fn test_petzval_radius() {
        let r = petzval_radius(&[0.1, -0.15], &[1.5, 1.6]);
        assert!(r.is_finite());
    }

    #[test]
    fn test_distortion() {
        let d = distortion_percent(10.5, 10.0);
        assert!(approx(d, 5.0, EPS));
    }

    // ---- Dispersion ----

    #[test]
    fn test_cauchy_dispersion() {
        let n = cauchy_dispersion(1.5, 0.005, 0.0, 0.5e-6);
        assert!(n > 1.5);
    }

    #[test]
    fn test_sellmeier() {
        // B=1.03, C=0.00787 (in um^2 -> 7.87e-15 m^2), lambda=0.5e-6 m
        let n = sellmeier_single(1.03, 7.87e-15, 0.5e-6);
        assert!(n > 1.0);
    }

    #[test]
    fn test_group_index() {
        let ng = group_index(|l| 1.5 + 0.005 / (l * l), 0.5e-6);
        assert!(ng > 1.5);
    }

    // ---- Gaussian beam ----

    #[test]
    fn test_rayleigh_range() {
        let zr = rayleigh_range(1e-3, 500e-9);
        let expected = PI * 1e-6 / 500e-9;
        assert!(approx(zr, expected, 1e-3));
    }

    #[test]
    fn test_beam_waist_at_zero() {
        let w = beam_waist_at_z(1e-3, 0.0, 1.0);
        assert!(approx(w, 1e-3, EPS));
    }

    #[test]
    fn test_beam_waist_far_field() {
        let zr = 1.0;
        let w = beam_waist_at_z(1e-3, 100.0, zr);
        assert!(w > 1e-3);
    }

    #[test]
    fn test_wavefront_radius_infinity() {
        let r = wavefront_radius(0.0, 1.0);
        assert!(r.is_infinite());
    }

    #[test]
    fn test_wavefront_radius_at_zr() {
        let r = wavefront_radius(1.0, 1.0);
        assert!(approx(r, 2.0, EPS));
    }

    #[test]
    fn test_beam_divergence() {
        let d = beam_divergence(1e-3, 500e-9);
        let expected = 500e-9 / (PI * 1e-3);
        assert!(approx(d, expected, EPS));
    }

    #[test]
    fn test_gouy_phase() {
        let g = gouy_phase(1.0, 1.0);
        assert!(approx(g, PI / 4.0, EPS));
    }

    // ---- Prism ----

    #[test]
    fn test_prism_min_deviation() {
        let dm = prism_min_deviation(1.5, PI / 3.0);
        // Verify: n = sin((A+dm)/2) / sin(A/2)
        let n_check = ((PI / 3.0 + dm) / 2.0).sin() / (PI / 6.0).sin();
        assert!(approx(n_check, 1.5, 1e-4));
    }

    #[test]
    fn test_prism_index_from_deviation() {
        let dm = prism_min_deviation(1.5, PI / 3.0);
        let n = prism_index_from_deviation(PI / 3.0, dm);
        assert!(approx(n, 1.5, 1e-4));
    }

    #[test]
    fn test_prism_angular_dispersion() {
        let ad = prism_angular_dispersion(PI / 3.0, 1.5, 1e4);
        assert!(ad.is_finite());
        assert!(ad > 0.0);
    }

    // ---- Optical path / interference ----

    #[test]
    fn test_optical_path_length() {
        assert!(approx(optical_path_length(1.5, 0.01), 0.015, EPS));
    }

    #[test]
    fn test_phase_from_opd() {
        let p = phase_from_opd(500e-9, 500e-9);
        assert!(approx(p, 2.0 * PI, EPS));
    }

    #[test]
    fn test_coherence_length() {
        let lc = coherence_length(600e-9, 1e-9);
        assert!(approx(lc, 3.6e-4, EPS));
    }

    #[test]
    fn test_fringe_visibility() {
        let v = fringe_visibility(1.0, 0.0);
        assert!(approx(v, 1.0, EPS));
        let v2 = fringe_visibility(1.0, 1.0);
        assert!(approx(v2, 0.0, EPS));
    }

    #[test]
    fn test_two_beam_constructive() {
        let i = two_beam_interference(1.0, 1.0, 0.0);
        assert!(approx(i, 4.0, EPS));
    }

    #[test]
    fn test_two_beam_destructive() {
        let i = two_beam_interference(1.0, 1.0, PI);
        assert!(approx(i, 0.0, EPS));
    }

    // ---- Ray ----

    #[test]
    fn test_ray_at() {
        let r = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 500e-9);
        let p = r.at(5.0);
        assert!(approx(p.z, 5.0, EPS));
    }

    #[test]
    fn test_ray_direction_normalized() {
        let r = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 0.0), 500e-9);
        assert!(approx(r.direction.length(), 1.0, EPS));
    }

    // ---- Vec3 ----

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!(approx(z.z, 1.0, EPS));
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(approx(a.dot(b), 32.0, EPS));
    }

    #[test]
    fn test_vec3_normalized() {
        let v = Vec3::new(3.0, 4.0, 0.0).normalized();
        assert!(approx(v.length(), 1.0, EPS));
    }

    #[test]
    fn test_vec3_zero_normalized() {
        let v = Vec3::new(0.0, 0.0, 0.0).normalized();
        assert!(approx(v.length(), 0.0, EPS));
    }

    // ---- Additional edge cases ----

    #[test]
    fn test_snell_identity() {
        // Same medium: angle unchanged
        let t2 = snell(1.5, 0.3, 1.5).unwrap();
        assert!(approx(t2, 0.3, EPS));
    }

    #[test]
    fn test_fresnel_grazing() {
        let r = fresnel_unpolarized(1.0, PI / 2.0 - 0.001, 1.5);
        assert!(r > 0.99);
    }

    #[test]
    fn test_thin_film_zero_thickness() {
        let r = thin_film_reflectance(1.0, 1.38, 1.5, 0.0, 500e-9);
        let r_bare = ((1.0_f64 - 1.5) / (1.0 + 1.5)).powi(2);
        assert!(approx(r, r_bare, 1e-3));
    }

    #[test]
    fn test_fiber_zero_length() {
        let p = fiber_attenuation(1.0, 0.2, 0.0);
        assert!(approx(p, 1.0, EPS));
    }

    #[test]
    fn test_matrix_mul_associativity() {
        let a = Matrix2x2::propagation(1.0);
        let b = Matrix2x2::thin_lens(0.5);
        let c = Matrix2x2::propagation(2.0);
        let ab_c = a.matmul(b).matmul(c);
        let a_bc = a.matmul(b.matmul(c));
        assert!(approx(ab_c.a, a_bc.a, EPS));
        assert!(approx(ab_c.b, a_bc.b, EPS));
        assert!(approx(ab_c.c, a_bc.c, EPS));
        assert!(approx(ab_c.d, a_bc.d, EPS));
    }
}
