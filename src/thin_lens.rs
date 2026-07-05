//! Thin lens equations.

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
