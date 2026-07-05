//! Diffraction: single/double slit, Airy disk, grating.

use core::f64::consts::PI;

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
