//! Gaussian beam optics.

use core::f64::consts::PI;

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
