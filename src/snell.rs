//! Snell's law.

use crate::vec3::Vec3;

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
