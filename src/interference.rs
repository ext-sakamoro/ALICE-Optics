//! Optical path length + two-beam interference.

use core::f64::consts::PI;

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
