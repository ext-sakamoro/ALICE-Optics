//! Thin film interference.

use core::f64::consts::PI;

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
