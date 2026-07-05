//! Optical aberrations: spherical / chromatic / coma / Petzval / distortion.

use crate::thin_lens::lensmaker_thin;

/// Longitudinal spherical aberration for a thin lens.
/// `f` = focal length, `h` = ray height, `n` = refractive index.
/// Uses third-order approximation.
#[must_use]
pub fn spherical_aberration_longitudinal(f: f64, h: f64, n: f64) -> f64 {
    let q = 0.0;
    let p = 0.0;
    let s_coeff = ((1.0 / (16.0 * n * (n - 1.0))) * (n + 2.0) / (n - 1.0)).mul_add(
        ((2.0 * n.mul_add(n, -1.0)) / (n + 2.0))
            .mul_add(q, p)
            .powi(2),
        n * n / ((n - 1.0) * (n - 1.0)),
    );
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
