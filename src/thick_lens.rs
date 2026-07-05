//! Thick lens equations.

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
