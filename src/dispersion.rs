//! Dispersion: Cauchy, Sellmeier, group index.

/// Cauchy dispersion formula: n(lambda) = a + b / lambda^2 + c / lambda^4.
#[must_use]
pub fn cauchy_dispersion(a: f64, b: f64, c: f64, lambda: f64) -> f64 {
    a + b / (lambda * lambda) + c / (lambda * lambda * lambda * lambda)
}

/// Sellmeier equation (single term): n^2 - 1 = B * lambda^2 / (lambda^2 - C).
#[must_use]
pub fn sellmeier_single(big_b: f64, big_c: f64, lambda: f64) -> f64 {
    (1.0 + big_b * lambda * lambda / lambda.mul_add(lambda, -big_c)).sqrt()
}

/// Group refractive index: `n_g` = n - lambda * dn/dlambda.
/// Uses numerical differentiation.
#[must_use]
pub fn group_index(n_func: fn(f64) -> f64, lambda: f64) -> f64 {
    let dl = lambda * 1e-6;
    let dn = (n_func(lambda + dl) - n_func(lambda - dl)) / (2.0 * dl);
    lambda.mul_add(-dn, n_func(lambda))
}
