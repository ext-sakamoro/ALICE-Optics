//! Fiber optics.

use core::f64::consts::PI;

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
