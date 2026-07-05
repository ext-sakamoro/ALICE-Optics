//! Fresnel reflection equations.

/// Fresnel reflectance for s-polarization.
#[must_use]
pub fn fresnel_rs(n1: f64, theta_i: f64, n2: f64) -> f64 {
    let cos_i = theta_i.cos();
    let sin_t = n1 * theta_i.sin() / n2;
    if sin_t.abs() > 1.0 {
        return 1.0;
    }
    let cos_t = sin_t.mul_add(-sin_t, 1.0).sqrt();
    let num = n1.mul_add(cos_i, -(n2 * cos_t));
    let den = n1.mul_add(cos_i, n2 * cos_t);
    (num / den) * (num / den)
}

/// Fresnel reflectance for p-polarization.
#[must_use]
pub fn fresnel_rp(n1: f64, theta_i: f64, n2: f64) -> f64 {
    let cos_i = theta_i.cos();
    let sin_t = n1 * theta_i.sin() / n2;
    if sin_t.abs() > 1.0 {
        return 1.0;
    }
    let cos_t = sin_t.mul_add(-sin_t, 1.0).sqrt();
    let num = n2.mul_add(cos_i, -(n1 * cos_t));
    let den = n2.mul_add(cos_i, n1 * cos_t);
    (num / den) * (num / den)
}

/// Unpolarized Fresnel reflectance (average of s and p).
#[must_use]
pub fn fresnel_unpolarized(n1: f64, theta_i: f64, n2: f64) -> f64 {
    0.5 * (fresnel_rs(n1, theta_i, n2) + fresnel_rp(n1, theta_i, n2))
}

/// Brewster's angle for interface n1 -> n2.
#[must_use]
pub fn brewster_angle(n1: f64, n2: f64) -> f64 {
    (n2 / n1).atan()
}

/// Critical angle for total internal reflection (n1 > n2).
/// Returns `None` if n1 <= n2.
#[must_use]
pub fn critical_angle(n1: f64, n2: f64) -> Option<f64> {
    if n1 <= n2 {
        None
    } else {
        Some((n2 / n1).asin())
    }
}
