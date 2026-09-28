//! Closed-form oracles for the caustic tracer.
//!
//! Every expectation here comes from optics written out by hand, never from
//! calling the tracer and pinning what it returned. The sources are named per
//! test. This file is an integration test, so it sees the crate exactly as a
//! downstream user does.

use alice_optics::caustics::{
    trace_caustics, CausticConfig, CausticError, SurfaceGrid, IOR_AIR, IOR_WATER,
};
use alice_optics::fresnel::fresnel_unpolarized;
use alice_optics::vec3::Vec3;

/// Relative comparison; `expected` is never zero in these tests.
fn assert_rel(actual: f64, expected: f64, tol: f64, what: &str) {
    let rel = (actual - expected).abs() / expected.abs();
    assert!(
        rel <= tol,
        "{what}: got {actual:.12e}, expected {expected:.12e}, relative error {rel:.3e} > {tol:.3e}"
    );
}

/// A flat surface at `y = 0` with exact upward normals.
fn flat_surface(n: usize) -> (Vec<f64>, Vec<Vec3>) {
    (vec![0.0; n * n], vec![Vec3::new(0.0, 1.0, 0.0); n * n])
}

/// A plane of constant slope `slope` along x, with its exact normal.
fn tilted_surface(n: usize, extent: f64, slope: f64) -> (Vec<f64>, Vec<Vec3>) {
    let mut heights = Vec::with_capacity(n * n);
    let mut normals = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let (x, _z) = node_xz(i, j, n, extent);
            heights.push(slope * x);
            // Surface y = slope * x has upward normal (-slope, 1, 0).
            normals.push(Vec3::new(-slope, 1.0, 0.0).normalized());
        }
    }
    (heights, normals)
}

/// A shallow parabolic dip `h = -(x^2 + z^2) / (2R)` with its exact normal.
///
/// For a height field `y = h(x, z)` the upward normal is proportional to
/// `(-dh/dx, 1, -dh/dz)`, which here is `(x/R, 1, z/R)`.
fn parabolic_dip(n: usize, extent: f64, radius: f64) -> (Vec<f64>, Vec<Vec3>) {
    let mut heights = Vec::with_capacity(n * n);
    let mut normals = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let (x, z) = node_xz(i, j, n, extent);
            heights.push(-(x * x + z * z) / (2.0 * radius));
            normals.push(Vec3::new(x / radius, 1.0, z / radius).normalized());
        }
    }
    (heights, normals)
}

/// World `(x, z)` of lattice node `(i, j)` on an `n` x `n` lattice of width `extent`.
fn node_xz(i: usize, j: usize, n: usize, extent: f64) -> (f64, f64) {
    let step = extent / (n - 1) as f64;
    let half = extent / 2.0;
    (
        (i as f64).mul_add(step, -half),
        (j as f64).mul_add(step, -half),
    )
}

/// Paraxial deflection factor for a surface between `n_above` and `n_below`.
///
/// Oracle: to first order in the surface slope `s`, a ray travelling straight down
/// leaves the interface tilted by `(1 - n_above/n_below) * s` from the vertical.
/// Derived from the vector form of Snell's law with `cos(theta) -> 1`.
fn deflection_factor(n_above: f64, n_below: f64) -> f64 {
    1.0 - n_above / n_below
}

// ---------------------------------------------------------------------------
// Oracle 1: an affine surface cannot concentrate light.
// ---------------------------------------------------------------------------

#[test]
fn flat_surface_under_vertical_light_has_unit_concentration_everywhere() {
    // Oracle: a flat interface refracts every ray into the same direction, so the
    // landing map is a pure translation. A translation has Jacobian determinant 1,
    // so no patch of floor is brighter than any other.
    let (heights, normals) = flat_surface(9);
    let grid = SurfaceGrid::new(9, 9, 4.0, 4.0, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(-3.0).unwrap();

    let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();

    for (index, s) in field.samples.iter().enumerate() {
        assert_rel(s.concentration, 1.0, 1e-12, &format!("node {index}"));
    }
}

#[test]
fn flat_surface_under_oblique_light_has_unit_concentration_everywhere() {
    // Same oracle: the direction changes but stays constant across the surface, so
    // the map is still a translation.
    let (heights, normals) = flat_surface(9);
    let grid = SurfaceGrid::new(9, 9, 4.0, 4.0, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(-3.0).unwrap();
    let light = Vec3::new(0.4, -1.0, 0.25).normalized();

    let field = trace_caustics(&grid, light, &cfg).unwrap();

    for (index, s) in field.samples.iter().enumerate() {
        assert_rel(s.concentration, 1.0, 1e-12, &format!("node {index}"));
    }
}

#[test]
fn uniformly_tilted_surface_has_unit_concentration_everywhere() {
    // Oracle: a constant slope means a constant normal, so again one refracted
    // direction for the whole surface and again a translation.
    let (heights, normals) = tilted_surface(9, 4.0, 0.2);
    let grid = SurfaceGrid::new(9, 9, 4.0, 4.0, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(-6.0).unwrap();

    let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();

    for (index, s) in field.samples.iter().enumerate() {
        assert_rel(s.concentration, 1.0, 1e-9, &format!("node {index}"));
    }
}

// ---------------------------------------------------------------------------
// Oracle 2: the radiometric factors, each against its own closed form.
// ---------------------------------------------------------------------------

#[test]
fn normal_incidence_transmittance_matches_closed_form() {
    // Oracle: at normal incidence the Fresnel reflectance is ((n1-n2)/(n1+n2))^2.
    // For n1 = 1 and n2 = 4/3 that is (1/7)^2 = 1/49, so 48/49 is transmitted, and
    // with the light straight down the obliquity factor is exactly 1.
    let (heights, normals) = flat_surface(5);
    let grid = SurfaceGrid::new(5, 5, 2.0, 2.0, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(-2.0).unwrap();

    let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();

    let expected = 48.0 / 49.0;
    for (index, s) in field.samples.iter().enumerate() {
        assert_rel(s.transmitted, expected, 1e-12, &format!("node {index}"));
    }
}

#[test]
fn oblique_transmittance_matches_the_fresnel_module() {
    // Oracle: an independent implementation already in this crate. The obliquity
    // factor is cos(theta) because the lattice footprint is measured horizontally.
    let (heights, normals) = flat_surface(5);
    let grid = SurfaceGrid::new(5, 5, 2.0, 2.0, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(-2.0).unwrap();

    let theta = 0.6_f64; // radians from the vertical
    let light = Vec3::new(theta.sin(), -theta.cos(), 0.0);
    let field = trace_caustics(&grid, light, &cfg).unwrap();

    let expected = theta.cos() * (1.0 - fresnel_unpolarized(IOR_AIR, theta, IOR_WATER));
    for (index, s) in field.samples.iter().enumerate() {
        assert_rel(s.transmitted, expected, 1e-12, &format!("node {index}"));
    }
}

#[test]
fn beer_lambert_attenuation_matches_closed_form() {
    // Oracle: straight down through a flat surface, the path to a floor at depth D
    // has length exactly D, so the surviving fraction is exp(-a D) on top of the
    // 48/49 that got through the interface.
    let depth = 2.5_f64;
    let absorption = 0.37_f64;
    let (heights, normals) = flat_surface(5);
    let grid = SurfaceGrid::new(5, 5, 2.0, 2.0, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(-depth)
        .unwrap()
        .with_absorption(absorption)
        .unwrap();

    let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();

    let expected = (48.0 / 49.0) * (-absorption * depth).exp();
    for (index, s) in field.samples.iter().enumerate() {
        assert_rel(s.transmitted, expected, 1e-12, &format!("node {index}"));
    }
}

// ---------------------------------------------------------------------------
// Oracle 3: a lens with a known focal length.
// ---------------------------------------------------------------------------

/// Concentration a paraxial parabolic dip produces at depth `d`.
///
/// Oracle: with `h = -(x^2+z^2)/(2R)` the slope at `x` is `-x/R`, so a ray landing
/// after depth `d` arrives at `x * (1 - k d / R)` with `k = 1 - n_above/n_below`.
/// That map is a uniform scale, so its Jacobian determinant is the square of that
/// factor and the concentration is its reciprocal.
fn paraxial_concentration(radius: f64, depth: f64, k: f64) -> f64 {
    let scale = k.mul_add(-depth / radius, 1.0);
    1.0 / (scale * scale)
}

#[test]
fn parabolic_dip_concentration_matches_paraxial_closed_form() {
    let radius = 200.0;
    let extent = 4.0; // |x| <= 2, so the slope stays below 0.01 and paraxial holds
    let n = 41;
    let k = deflection_factor(IOR_AIR, IOR_WATER);
    assert_rel(k, 0.25, 1e-15, "deflection factor for n = 4/3");

    for depth in [50.0_f64, 100.0, 400.0] {
        let (heights, normals) = parabolic_dip(n, extent, radius);
        let grid = SurfaceGrid::new(n, n, extent, extent, &heights, &normals).unwrap();
        let cfg = CausticConfig::preset_air_to_water(-depth).unwrap();

        let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();

        let expected = paraxial_concentration(radius, depth, k);
        let centre = field.at(n / 2, n / 2);
        assert_rel(
            centre.concentration,
            expected,
            2e-3,
            &format!("centre concentration at depth {depth}"),
        );
    }
}

#[test]
fn parabolic_dip_focus_is_at_the_closed_form_depth() {
    // Oracle: the scale factor 1 - k d / R vanishes at d = R / k, which for water
    // (k = 1/4) is four radii down. Approaching it, the concentration must grow
    // like 1 / (1 - k d / R)^2 and exceed any fixed bound.
    let radius = 200.0;
    let extent = 4.0;
    let n = 41;
    let k = deflection_factor(IOR_AIR, IOR_WATER);
    let focal_depth = radius / k;
    assert_rel(focal_depth, 800.0, 1e-15, "focal depth for n = 4/3");

    let (heights, normals) = parabolic_dip(n, extent, radius);
    let grid = SurfaceGrid::new(n, n, extent, extent, &heights, &normals).unwrap();

    let mut previous = 0.0_f64;
    for fraction in [0.5_f64, 0.75, 0.9] {
        let depth = focal_depth * fraction;
        let cfg = CausticConfig::preset_air_to_water(-depth).unwrap();
        let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();
        let got = field.at(n / 2, n / 2).concentration;

        assert_rel(
            got,
            paraxial_concentration(radius, depth, k),
            1e-2,
            &format!("concentration at {fraction} of the focal depth"),
        );
        assert!(
            got > previous,
            "concentration must grow toward the focus: {got} after {previous}"
        );
        previous = got;
    }
    assert!(
        previous > 50.0,
        "at 0.9 of the focal depth the closed form is 100, got {previous}"
    );
}

// ---------------------------------------------------------------------------
// Oracle 4: the Jacobian against geometry, and the energy it implies.
// ---------------------------------------------------------------------------

/// Area of the quadrilateral `a b c d` in the xz plane, by the shoelace formula.
fn quad_area_xz(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    let cross = |p: Vec3, q: Vec3| p.x.mul_add(q.z, -(p.z * q.x));
    0.5 * (cross(a, b) + cross(b, c) + cross(c, d) + cross(d, a)).abs()
}

#[test]
fn jacobian_concentration_matches_the_landed_quad_area() {
    // Oracle: the concentration is defined as a ratio of areas, so measuring the
    // landed quadrilateral directly must reproduce it. This routes around the
    // derivative entirely: the tracer computes it from a difference quotient of
    // the landing map, the test computes it from the corners it actually returned.
    let radius = 200.0;
    let extent = 4.0;
    let n = 41;
    let depth = 300.0;
    let (heights, normals) = parabolic_dip(n, extent, radius);
    let grid = SurfaceGrid::new(n, n, extent, extent, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(-depth).unwrap();

    let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();

    // Interior cells only: the tracer has one-sided derivatives on the border.
    for j in 1..n - 2 {
        for i in 1..n - 2 {
            let landed = quad_area_xz(
                field.at(i, j).landing,
                field.at(i + 1, j).landing,
                field.at(i + 1, j + 1).landing,
                field.at(i, j + 1).landing,
            );
            let geometric = field.cell_area / landed;
            let average = 0.25
                * (field.at(i, j).concentration
                    + field.at(i + 1, j).concentration
                    + field.at(i + 1, j + 1).concentration
                    + field.at(i, j + 1).concentration);
            assert_rel(average, geometric, 1e-6, &format!("cell ({i}, {j})"));
        }
    }
}

#[test]
fn total_flux_is_conserved_between_the_surface_and_the_floor() {
    // Oracle: refraction moves light, it does not create or destroy it. With the
    // absorption set to zero, the power crossing the interface must equal the power
    // landing on the floor. The two sums are built along different routes: the left
    // from the lattice footprint, the right from the quadrilaterals the rays landed
    // on.
    let radius = 200.0;
    let extent = 4.0;
    let n = 41;
    let depth = 300.0;
    let (heights, normals) = parabolic_dip(n, extent, radius);
    let grid = SurfaceGrid::new(n, n, extent, extent, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(-depth).unwrap();

    let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();

    let mut leaving = 0.0_f64;
    let mut landing = 0.0_f64;
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let corners = [
                field.at(i, j),
                field.at(i + 1, j),
                field.at(i + 1, j + 1),
                field.at(i, j + 1),
            ];
            let mean_transmitted = 0.25 * corners.iter().map(|s| s.transmitted).sum::<f64>();
            let mean_floor = 0.25 * corners.iter().map(|s| s.floor_irradiance()).sum::<f64>();
            let landed_area = quad_area_xz(
                corners[0].landing,
                corners[1].landing,
                corners[2].landing,
                corners[3].landing,
            );
            leaving += mean_transmitted * field.cell_area;
            landing += mean_floor * landed_area;
        }
    }

    assert_rel(landing, leaving, 1e-6, "total flux");
}

// ---------------------------------------------------------------------------
// Oracle 5: the answer must not depend on how finely the surface was sampled.
// ---------------------------------------------------------------------------

#[test]
fn concentration_is_independent_of_lattice_resolution() {
    // The lattice is a sampling of the surface, not part of the physics, so
    // refining it must leave the answer alone.
    let radius = 200.0;
    let extent = 4.0;
    let depth = 300.0;
    let expected = paraxial_concentration(radius, depth, deflection_factor(IOR_AIR, IOR_WATER));

    for n in [11_usize, 21, 41, 81] {
        let (heights, normals) = parabolic_dip(n, extent, radius);
        let grid = SurfaceGrid::new(n, n, extent, extent, &heights, &normals).unwrap();
        let cfg = CausticConfig::preset_air_to_water(-depth).unwrap();

        let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();

        assert_rel(
            field.at(n / 2, n / 2).concentration,
            expected,
            5e-3,
            &format!("centre concentration at resolution {n}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Rejections: every one of these must fail loudly rather than return a number.
// ---------------------------------------------------------------------------

#[test]
fn a_lattice_without_a_spacing_is_rejected() {
    let heights = vec![0.0; 1];
    let normals = vec![Vec3::new(0.0, 1.0, 0.0); 1];
    assert_eq!(
        SurfaceGrid::new(1, 1, 1.0, 1.0, &heights, &normals).unwrap_err(),
        CausticError::GridTooSmall
    );
}

#[test]
fn mismatched_slice_lengths_are_rejected() {
    let heights = vec![0.0; 8];
    let normals = vec![Vec3::new(0.0, 1.0, 0.0); 9];
    assert_eq!(
        SurfaceGrid::new(3, 3, 1.0, 1.0, &heights, &normals).unwrap_err(),
        CausticError::GridLengthMismatch
    );
}

#[test]
fn a_non_positive_extent_is_rejected() {
    let (heights, normals) = flat_surface(3);
    assert_eq!(
        SurfaceGrid::new(3, 3, 0.0, 1.0, &heights, &normals).unwrap_err(),
        CausticError::GridExtentInvalid
    );
}

#[test]
fn a_refractive_index_below_one_is_rejected() {
    assert_eq!(
        CausticConfig::try_new(1.0, 0.5, -1.0, 0.0, 1.0).unwrap_err(),
        CausticError::IorInvalid
    );
}

#[test]
fn negative_absorption_is_rejected() {
    assert_eq!(
        CausticConfig::try_new(1.0, IOR_WATER, -1.0, -0.1, 1.0).unwrap_err(),
        CausticError::AbsorptionInvalid
    );
}

#[test]
fn non_positive_source_irradiance_is_rejected() {
    assert_eq!(
        CausticConfig::try_new(1.0, IOR_WATER, -1.0, 0.0, 0.0).unwrap_err(),
        CausticError::IrradianceInvalid
    );
}

#[test]
fn light_that_does_not_travel_downward_is_rejected() {
    let (heights, normals) = flat_surface(3);
    let grid = SurfaceGrid::new(3, 3, 1.0, 1.0, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(-1.0).unwrap();
    assert_eq!(
        trace_caustics(&grid, Vec3::new(0.0, 1.0, 0.0), &cfg).unwrap_err(),
        CausticError::LightDirectionInvalid
    );
    assert_eq!(
        trace_caustics(&grid, Vec3::new(0.0, 0.0, 0.0), &cfg).unwrap_err(),
        CausticError::LightDirectionInvalid
    );
}

#[test]
fn a_floor_above_the_surface_is_rejected() {
    let (heights, normals) = flat_surface(3);
    let grid = SurfaceGrid::new(3, 3, 1.0, 1.0, &heights, &normals).unwrap();
    let cfg = CausticConfig::preset_air_to_water(0.5).unwrap();
    assert!(matches!(
        trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap_err(),
        CausticError::FloorAboveSurface { .. }
    ));
}
