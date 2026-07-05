//! Prism deviation and dispersion.

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
