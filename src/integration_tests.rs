//! Integration tests moved from monolithic lib.rs.

#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::redundant_clone,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::manual_midpoint
)]

use core::f64::consts::PI;

use crate::aberrations::{
    abbe_number, achromatic_doublet, chromatic_aberration, coma_coefficient, distortion_percent,
    petzval_radius, spherical_aberration_longitudinal, spherical_aberration_transverse,
};
use crate::diffraction::{
    airy_first_zero, double_slit_intensity, double_slit_maxima, grating_maxima,
    rayleigh_resolution, single_slit_intensity, single_slit_minima,
};
use crate::dispersion::{cauchy_dispersion, group_index, sellmeier_single};
use crate::fiber::{
    fiber_acceptance_angle, fiber_attenuation, fiber_is_single_mode, fiber_na, fiber_num_modes,
    fiber_v_number, modal_dispersion,
};
use crate::fresnel::{brewster_angle, critical_angle, fresnel_rp, fresnel_rs, fresnel_unpolarized};
use crate::gaussian_beam::{
    beam_divergence, beam_waist_at_z, gouy_phase, rayleigh_range, wavefront_radius,
};
use crate::interference::{
    coherence_length, fringe_visibility, optical_path_length, phase_from_opd, two_beam_interference,
};
use crate::jones::{JonesMatrix, JonesVector};
use crate::matrix_optics::{compose_matrices, trace_paraxial, Matrix2x2, SphericalSurface};
use crate::prism::{prism_angular_dispersion, prism_index_from_deviation, prism_min_deviation};
use crate::ray::Ray;
use crate::snell::{snell, snell_vec};
use crate::stokes::{MuellerMatrix, StokesVector};
use crate::thick_lens::{
    thick_lens_back_principal, thick_lens_focal_length, thick_lens_front_principal,
};
use crate::thin_film::{
    anti_reflection_index, anti_reflection_thickness, thin_film_constructive_lambda,
    thin_film_destructive_lambda, thin_film_reflectance,
};
use crate::thin_lens::{
    lensmaker_thin, optical_power, thin_lens_image_distance, thin_lens_magnification,
};
use crate::vec3::Vec3;

const EPS: f64 = 1e-6;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ---- Snell's law ----

#[test]
fn test_snell_normal_incidence() {
    let t2 = snell(1.0, 0.0, 1.5).unwrap();
    assert!(approx(t2, 0.0, EPS));
}

#[test]
fn test_snell_known_angle() {
    let t2 = snell(1.0, PI / 6.0, 1.5).unwrap();
    let expected = (0.5 / 1.5_f64).asin();
    assert!(approx(t2, expected, EPS));
}

#[test]
fn test_snell_tir() {
    assert!(snell(1.5, PI / 3.0, 1.0).is_none());
}

#[test]
fn test_snell_vec_normal() {
    let d = snell_vec(
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
        1.0,
        1.5,
    );
    assert!(d.is_some());
    let r = d.unwrap();
    assert!(approx(r.x, 0.0, EPS));
    assert!(approx(r.y, 0.0, EPS));
    assert!(r.z < 0.0);
}

#[test]
fn test_snell_vec_tir() {
    let d = snell_vec(
        Vec3::new(0.9, 0.0, -0.4).normalized(),
        Vec3::new(0.0, 0.0, 1.0),
        1.5,
        1.0,
    );
    assert!(d.is_none());
}

// ---- Fresnel equations ----

#[test]
fn test_fresnel_normal_incidence() {
    let r = fresnel_unpolarized(1.0, 0.0, 1.5);
    let expected = ((1.0_f64 - 1.5) / (1.0 + 1.5)).powi(2);
    assert!(approx(r, expected, EPS));
}

#[test]
fn test_fresnel_rp_at_brewster() {
    let b = brewster_angle(1.0, 1.5);
    let rp = fresnel_rp(1.0, b, 1.5);
    assert!(approx(rp, 0.0, 1e-4));
}

#[test]
fn test_fresnel_tir() {
    let r = fresnel_rs(1.5, PI / 3.0, 1.0);
    assert!(approx(r, 1.0, EPS));
}

#[test]
fn test_brewster_angle() {
    let b = brewster_angle(1.0, 1.5);
    assert!(approx(b.tan(), 1.5, EPS));
}

#[test]
fn test_critical_angle() {
    let c = critical_angle(1.5, 1.0).unwrap();
    assert!(approx(c.sin(), 1.0 / 1.5, EPS));
}

#[test]
fn test_critical_angle_none() {
    assert!(critical_angle(1.0, 1.5).is_none());
}

// ---- Thin lens ----

#[test]
fn test_thin_lens_image_at_2f() {
    let d_i = thin_lens_image_distance(0.1, 0.2);
    assert!(approx(d_i, 0.2, EPS));
}

#[test]
fn test_thin_lens_image_at_infinity() {
    let d_i = thin_lens_image_distance(0.1, 0.1);
    assert!(d_i > 1e10);
}

#[test]
fn test_thin_lens_magnification() {
    let m = thin_lens_magnification(0.2, 0.2);
    assert!(approx(m, -1.0, EPS));
}

#[test]
fn test_lensmaker_thin() {
    let f = lensmaker_thin(1.5, 0.2, -0.2);
    let expected = 1.0 / (0.5 * (1.0 / 0.2 + 1.0 / 0.2));
    assert!(approx(f, expected, EPS));
}

#[test]
fn test_optical_power() {
    assert!(approx(optical_power(0.5), 2.0, EPS));
}

// ---- Thick lens ----

#[test]
fn test_thick_lens_reduces_to_thin() {
    let f_thick = thick_lens_focal_length(1.5, 0.2, -0.2, 0.001);
    let f_thin = lensmaker_thin(1.5, 0.2, -0.2);
    assert!(approx(f_thick, f_thin, 1e-3));
}

#[test]
fn test_thick_lens_principal_planes() {
    let h1 = thick_lens_front_principal(1.5, 0.1, -0.1, 0.02);
    let h2 = thick_lens_back_principal(1.5, 0.1, -0.1, 0.02);
    assert!(approx(h1.abs(), h2.abs(), 1e-4));
}

// ---- Matrix optics ----

#[test]
fn test_propagation_matrix() {
    let m = Matrix2x2::propagation(1.0);
    let (y, theta) = m.apply(0.0, 0.1);
    assert!(approx(y, 0.1, EPS));
    assert!(approx(theta, 0.1, EPS));
}

#[test]
fn test_thin_lens_matrix() {
    let m = Matrix2x2::thin_lens(0.5);
    let (y, theta) = m.apply(0.1, 0.0);
    assert!(approx(y, 0.1, EPS));
    assert!(approx(theta, -0.2, EPS));
}

#[test]
fn test_matrix_det() {
    let m = Matrix2x2::propagation(2.0);
    assert!(approx(m.det(), 1.0, EPS));
}

#[test]
fn test_matrix_identity() {
    let m = Matrix2x2::identity();
    let (y, t) = m.apply(3.0, 0.5);
    assert!(approx(y, 3.0, EPS));
    assert!(approx(t, 0.5, EPS));
}

#[test]
fn test_compose_matrices() {
    let m = compose_matrices(&[
        Matrix2x2::propagation(1.0),
        Matrix2x2::thin_lens(0.5),
        Matrix2x2::propagation(1.0),
    ]);
    let (y, _) = m.apply(0.1, 0.0);
    assert!(y.is_finite());
}

#[test]
fn test_flat_refraction() {
    let m = Matrix2x2::flat_refraction(1.0, 1.5);
    assert!(approx(m.d, 1.0 / 1.5, EPS));
}

#[test]
fn test_curved_refraction() {
    let m = Matrix2x2::curved_refraction(1.0, 1.5, 0.1);
    assert!(approx(m.c, (1.0 - 1.5) / (1.5 * 0.1), EPS));
}

// ---- Paraxial ray tracing ----

#[test]
fn test_trace_paraxial_single_surface() {
    let surfaces = [SphericalSurface {
        z: 0.0,
        radius: 0.1,
        n_after: 1.5,
    }];
    let (y, u) = trace_paraxial(&surfaces, 1.0, 0.01, 0.0);
    assert!(y.is_finite());
    assert!(u.is_finite());
}

#[test]
fn test_trace_paraxial_flat() {
    let surfaces = [SphericalSurface {
        z: 1.0,
        radius: 1e10,
        n_after: 1.0,
    }];
    let (y, u) = trace_paraxial(&surfaces, 1.0, 0.0, 0.1);
    assert!(approx(y, 0.1, 1e-3));
    assert!(approx(u, 0.1, 1e-3));
}

// ---- Single-slit diffraction ----

#[test]
fn test_single_slit_center() {
    let i = single_slit_intensity(1e-4, 500e-9, 0.0);
    assert!(approx(i, 1.0, EPS));
}

#[test]
fn test_single_slit_first_min() {
    let theta = (500e-9 / 1e-4_f64).asin();
    let i = single_slit_intensity(1e-4, 500e-9, theta);
    assert!(approx(i, 0.0, 1e-4));
}

#[test]
fn test_single_slit_symmetry() {
    let a = single_slit_intensity(1e-4, 500e-9, 0.001);
    let b = single_slit_intensity(1e-4, 500e-9, -0.001);
    assert!(approx(a, b, EPS));
}

// ---- Double-slit diffraction ----

#[test]
fn test_double_slit_center() {
    let i = double_slit_intensity(1e-5, 1e-4, 500e-9, 0.0);
    assert!(approx(i, 1.0, EPS));
}

#[test]
fn test_double_slit_envelope() {
    let theta = 0.001;
    let ds = double_slit_intensity(1e-5, 1e-4, 500e-9, theta);
    let ss = single_slit_intensity(1e-5, 500e-9, theta);
    assert!(ds <= ss + EPS);
}

#[test]
fn test_single_slit_minima() {
    let minima = single_slit_minima(1e-4, 500e-9, 3);
    assert_eq!(minima.len(), 3);
    for (m, angle) in minima.iter().enumerate() {
        let expected = ((m as f64 + 1.0) * 500e-9 / 1e-4).asin();
        assert!(approx(*angle, expected, EPS));
    }
}

#[test]
fn test_double_slit_maxima() {
    let maxima = double_slit_maxima(1e-4, 500e-9, 2);
    assert_eq!(maxima.len(), 3);
    assert!(approx(maxima[0], 0.0, EPS));
}

#[test]
fn test_airy_disk() {
    let theta = airy_first_zero(500e-9, 0.01);
    assert!(theta > 0.0);
}

#[test]
fn test_rayleigh_resolution() {
    let r = rayleigh_resolution(500e-9, 0.1);
    assert!(approx(r, 1.22 * 500e-9 / 0.1, EPS));
}

#[test]
fn test_grating_maxima() {
    let angles = grating_maxima(1e-5, 500e-9, 3);
    assert!(!angles.is_empty());
    assert!(angles.iter().any(|a| approx(*a, 0.0, EPS)));
}

// ---- Jones vectors and matrices ----

#[test]
fn test_jones_horizontal_intensity() {
    let h = JonesVector::horizontal();
    assert!(approx(h.intensity(), 1.0, EPS));
}

#[test]
fn test_jones_vertical_intensity() {
    let v = JonesVector::vertical();
    assert!(approx(v.intensity(), 1.0, EPS));
}

#[test]
fn test_jones_diagonal_intensity() {
    let d = JonesVector::diagonal();
    assert!(approx(d.intensity(), 1.0, EPS));
}

#[test]
fn test_jones_circular_intensity() {
    let r = JonesVector::right_circular();
    assert!(approx(r.intensity(), 1.0, EPS));
    let l = JonesVector::left_circular();
    assert!(approx(l.intensity(), 1.0, EPS));
}

#[test]
fn test_jones_polarizer_h_blocks_v() {
    let pol = JonesMatrix::polarizer_h();
    let v = JonesVector::vertical();
    let result = pol.apply(v);
    assert!(approx(result.intensity(), 0.0, EPS));
}

#[test]
fn test_jones_polarizer_h_passes_h() {
    let pol = JonesMatrix::polarizer_h();
    let h = JonesVector::horizontal();
    let result = pol.apply(h);
    assert!(approx(result.intensity(), 1.0, EPS));
}

#[test]
fn test_jones_polarizer_v_blocks_h() {
    let pol = JonesMatrix::polarizer_v();
    let h = JonesVector::horizontal();
    let result = pol.apply(h);
    assert!(approx(result.intensity(), 0.0, EPS));
}

#[test]
fn test_jones_crossed_polarizers() {
    let h_pol = JonesMatrix::polarizer_h();
    let v_pol = JonesMatrix::polarizer_v();
    let system = v_pol.matmul(h_pol);
    let light = JonesVector::diagonal();
    let result = system.apply(light);
    assert!(approx(result.intensity(), 0.0, EPS));
}

#[test]
fn test_jones_polarizer_45() {
    let pol = JonesMatrix::polarizer(PI / 4.0);
    let h = JonesVector::horizontal();
    let result = pol.apply(h);
    assert!(approx(result.intensity(), 0.5, EPS));
}

#[test]
fn test_jones_half_wave_plate() {
    let hwp = JonesMatrix::half_wave_plate(0.0);
    let h = JonesVector::horizontal();
    let result = hwp.apply(h);
    assert!(approx(result.intensity(), 1.0, EPS));
}

#[test]
fn test_jones_quarter_wave_plate() {
    let qwp = JonesMatrix::quarter_wave_plate(0.0);
    let h = JonesVector::horizontal();
    let result = qwp.apply(h);
    assert!(approx(result.intensity(), 1.0, EPS));
}

#[test]
fn test_jones_malus_law() {
    for deg in 0..=9 {
        let theta = f64::from(deg) * 10.0 * PI / 180.0;
        let pol = JonesMatrix::polarizer(theta);
        let h = JonesVector::horizontal();
        let result = pol.apply(h);
        let expected = theta.cos().powi(2);
        assert!(approx(result.intensity(), expected, 1e-4));
    }
}

// ---- Stokes vectors ----

#[test]
fn test_stokes_unpolarized_dop() {
    let s = StokesVector::unpolarized(1.0);
    assert!(approx(s.degree_of_polarization(), 0.0, EPS));
}

#[test]
fn test_stokes_horizontal_dop() {
    let s = StokesVector::horizontal(1.0);
    assert!(approx(s.degree_of_polarization(), 1.0, EPS));
}

#[test]
fn test_stokes_circular_dop() {
    let s = StokesVector::right_circular(1.0);
    assert!(approx(s.degree_of_polarization(), 1.0, EPS));
}

#[test]
fn test_stokes_orientation_horizontal() {
    let s = StokesVector::horizontal(1.0);
    assert!(approx(s.orientation_angle(), 0.0, EPS));
}

#[test]
fn test_stokes_orientation_vertical() {
    let s = StokesVector::vertical(1.0);
    assert!(approx(s.orientation_angle(), PI / 2.0, EPS));
}

#[test]
fn test_stokes_ellipticity_circular() {
    let s = StokesVector::right_circular(1.0);
    assert!(approx(s.ellipticity_angle(), PI / 4.0, EPS));
}

#[test]
fn test_stokes_ellipticity_linear() {
    let s = StokesVector::horizontal(1.0);
    assert!(approx(s.ellipticity_angle(), 0.0, EPS));
}

// ---- Mueller matrices ----

#[test]
fn test_mueller_identity() {
    let m = MuellerMatrix::identity();
    let s = StokesVector::horizontal(1.0);
    let result = m.apply(s);
    assert!(approx(result.s0, s.s0, EPS));
    assert!(approx(result.s1, s.s1, EPS));
}

#[test]
fn test_mueller_h_polarizer() {
    let m = MuellerMatrix::polarizer_h();
    let s = StokesVector::unpolarized(1.0);
    let result = m.apply(s);
    assert!(approx(result.s0, 0.5, EPS));
    assert!(approx(result.s1, 0.5, EPS));
}

#[test]
fn test_mueller_v_blocks_h() {
    let m = MuellerMatrix::polarizer_v();
    let s = StokesVector::horizontal(1.0);
    let result = m.apply(s);
    assert!(approx(result.s0, 0.0, EPS));
}

// ---- Thin film interference ----

#[test]
fn test_thin_film_quarter_wave_ar() {
    let n_film = (1.0_f64 * 1.5_f64).sqrt();
    let d = 500e-9 / (4.0 * n_film);
    let r = thin_film_reflectance(1.0, n_film, 1.5, d, 500e-9);
    assert!(approx(r, 0.0, 1e-4));
}

#[test]
fn test_thin_film_half_wave() {
    let n_film = 1.38;
    let d = 500e-9 / (2.0 * n_film);
    let r = thin_film_reflectance(1.0, n_film, 1.5, d, 500e-9);
    let r_bare = ((1.0_f64 - 1.5) / (1.0 + 1.5)).powi(2);
    assert!(approx(r, r_bare, 1e-3));
}

#[test]
fn test_anti_reflection_thickness() {
    let t = anti_reflection_thickness(1.38, 550e-9);
    assert!(approx(t, 550e-9 / (4.0 * 1.38), EPS));
}

#[test]
fn test_anti_reflection_index() {
    let n = anti_reflection_index(1.0, 1.5);
    assert!(approx(n, 1.5_f64.sqrt(), EPS));
}

#[test]
fn test_constructive_lambda() {
    let lam = thin_film_constructive_lambda(1.5, 200e-9, 1);
    assert!(approx(lam, 600e-9, EPS));
}

#[test]
fn test_destructive_lambda() {
    let lam = thin_film_destructive_lambda(1.5, 200e-9, 0);
    assert!(approx(lam, 600e-9 / 0.5, EPS));
}

// ---- Fiber optics ----

#[test]
fn test_fiber_na() {
    let na = fiber_na(1.48, 1.46);
    let expected = (1.48_f64.powi(2) - 1.46_f64.powi(2)).sqrt();
    assert!(approx(na, expected, EPS));
}

#[test]
fn test_fiber_acceptance_angle() {
    let a = fiber_acceptance_angle(1.48, 1.46);
    assert!(a > 0.0 && a < PI / 2.0);
}

#[test]
fn test_fiber_v_number() {
    let v = fiber_v_number(1.48, 1.46, 4e-6, 1.3e-6);
    assert!(v > 0.0);
}

#[test]
fn test_fiber_single_mode() {
    assert!(fiber_is_single_mode(2.0));
    assert!(!fiber_is_single_mode(3.0));
}

#[test]
fn test_fiber_num_modes() {
    assert!(approx(fiber_num_modes(10.0), 50.0, EPS));
}

#[test]
fn test_fiber_attenuation() {
    let p_out = fiber_attenuation(1.0, 0.2, 10.0);
    let expected = 10.0_f64.powf(-0.2);
    assert!(approx(p_out, expected, EPS));
}

#[test]
fn test_modal_dispersion() {
    let dt = modal_dispersion(1.48, 1.46, 1000.0);
    assert!(dt > 0.0);
}

// ---- Aberrations ----

#[test]
fn test_spherical_aberration() {
    let sa = spherical_aberration_longitudinal(0.1, 0.01, 1.5);
    assert!(sa > 0.0);
}

#[test]
fn test_transverse_aberration() {
    let ta = spherical_aberration_transverse(0.001, 0.1);
    assert!(approx(ta, 0.001 * 0.1_f64.tan(), EPS));
}

#[test]
fn test_chromatic_aberration() {
    let ca = chromatic_aberration(1.5, 0.005, 0.1, -0.1, 486e-9, 656e-9);
    assert!(ca.abs() > 0.0);
}

#[test]
fn test_abbe_number() {
    let v = abbe_number(1.5168, 1.5224, 1.5143);
    let expected = (1.5168 - 1.0) / (1.5224 - 1.5143);
    assert!(approx(v, expected, EPS));
}

#[test]
fn test_achromatic_doublet() {
    let (p1, p2) = achromatic_doublet(10.0, 64.0, 28.0);
    assert!(approx(p1 / 64.0 + p2 / 28.0, 0.0, 1e-10));
    assert!(approx(p1 + p2, 10.0, EPS));
}

#[test]
fn test_coma() {
    let c = coma_coefficient(0.1, 0.01, 1.5);
    assert!(c > 0.0);
}

#[test]
fn test_petzval_radius() {
    let r = petzval_radius(&[0.1, -0.15], &[1.5, 1.6]);
    assert!(r.is_finite());
}

#[test]
fn test_distortion() {
    let d = distortion_percent(10.5, 10.0);
    assert!(approx(d, 5.0, EPS));
}

// ---- Dispersion ----

#[test]
fn test_cauchy_dispersion() {
    let n = cauchy_dispersion(1.5, 0.005, 0.0, 0.5e-6);
    assert!(n > 1.5);
}

#[test]
fn test_sellmeier() {
    let n = sellmeier_single(1.03, 7.87e-15, 0.5e-6);
    assert!(n > 1.0);
}

#[test]
fn test_group_index() {
    let ng = group_index(|l| 1.5 + 0.005 / (l * l), 0.5e-6);
    assert!(ng > 1.5);
}

// ---- Gaussian beam ----

#[test]
fn test_rayleigh_range() {
    let zr = rayleigh_range(1e-3, 500e-9);
    let expected = PI * 1e-6 / 500e-9;
    assert!(approx(zr, expected, 1e-3));
}

#[test]
fn test_beam_waist_at_zero() {
    let w = beam_waist_at_z(1e-3, 0.0, 1.0);
    assert!(approx(w, 1e-3, EPS));
}

#[test]
fn test_beam_waist_far_field() {
    let zr = 1.0;
    let w = beam_waist_at_z(1e-3, 100.0, zr);
    assert!(w > 1e-3);
}

#[test]
fn test_wavefront_radius_infinity() {
    let r = wavefront_radius(0.0, 1.0);
    assert!(r.is_infinite());
}

#[test]
fn test_wavefront_radius_at_zr() {
    let r = wavefront_radius(1.0, 1.0);
    assert!(approx(r, 2.0, EPS));
}

#[test]
fn test_beam_divergence() {
    let d = beam_divergence(1e-3, 500e-9);
    let expected = 500e-9 / (PI * 1e-3);
    assert!(approx(d, expected, EPS));
}

#[test]
fn test_gouy_phase() {
    let g = gouy_phase(1.0, 1.0);
    assert!(approx(g, PI / 4.0, EPS));
}

// ---- Prism ----

#[test]
fn test_prism_min_deviation() {
    let dm = prism_min_deviation(1.5, PI / 3.0);
    let n_check = ((PI / 3.0 + dm) / 2.0).sin() / (PI / 6.0).sin();
    assert!(approx(n_check, 1.5, 1e-4));
}

#[test]
fn test_prism_index_from_deviation() {
    let dm = prism_min_deviation(1.5, PI / 3.0);
    let n = prism_index_from_deviation(PI / 3.0, dm);
    assert!(approx(n, 1.5, 1e-4));
}

#[test]
fn test_prism_angular_dispersion() {
    let ad = prism_angular_dispersion(PI / 3.0, 1.5, 1e4);
    assert!(ad.is_finite());
    assert!(ad > 0.0);
}

// ---- Optical path / interference ----

#[test]
fn test_optical_path_length() {
    assert!(approx(optical_path_length(1.5, 0.01), 0.015, EPS));
}

#[test]
fn test_phase_from_opd() {
    let p = phase_from_opd(500e-9, 500e-9);
    assert!(approx(p, 2.0 * PI, EPS));
}

#[test]
fn test_coherence_length() {
    let lc = coherence_length(600e-9, 1e-9);
    assert!(approx(lc, 3.6e-4, EPS));
}

#[test]
fn test_fringe_visibility() {
    let v = fringe_visibility(1.0, 0.0);
    assert!(approx(v, 1.0, EPS));
    let v2 = fringe_visibility(1.0, 1.0);
    assert!(approx(v2, 0.0, EPS));
}

#[test]
fn test_two_beam_constructive() {
    let i = two_beam_interference(1.0, 1.0, 0.0);
    assert!(approx(i, 4.0, EPS));
}

#[test]
fn test_two_beam_destructive() {
    let i = two_beam_interference(1.0, 1.0, PI);
    assert!(approx(i, 0.0, EPS));
}

// ---- Ray ----

#[test]
fn test_ray_at() {
    let r = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 500e-9);
    let p = r.at(5.0);
    assert!(approx(p.z, 5.0, EPS));
}

#[test]
fn test_ray_direction_normalized() {
    let r = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 0.0), 500e-9);
    assert!(approx(r.direction.length(), 1.0, EPS));
}

// ---- Vec3 ----

#[test]
fn test_vec3_cross() {
    let x = Vec3::new(1.0, 0.0, 0.0);
    let y = Vec3::new(0.0, 1.0, 0.0);
    let z = x.cross(y);
    assert!(approx(z.z, 1.0, EPS));
}

#[test]
fn test_vec3_dot() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    assert!(approx(a.dot(b), 32.0, EPS));
}

#[test]
fn test_vec3_normalized() {
    let v = Vec3::new(3.0, 4.0, 0.0).normalized();
    assert!(approx(v.length(), 1.0, EPS));
}

#[test]
fn test_vec3_zero_normalized() {
    let v = Vec3::new(0.0, 0.0, 0.0).normalized();
    assert!(approx(v.length(), 0.0, EPS));
}

// ---- Additional edge cases ----

#[test]
fn test_snell_identity() {
    let t2 = snell(1.5, 0.3, 1.5).unwrap();
    assert!(approx(t2, 0.3, EPS));
}

#[test]
fn test_fresnel_grazing() {
    let r = fresnel_unpolarized(1.0, PI / 2.0 - 0.001, 1.5);
    assert!(r > 0.99);
}

#[test]
fn test_thin_film_zero_thickness() {
    let r = thin_film_reflectance(1.0, 1.38, 1.5, 0.0, 500e-9);
    let r_bare = ((1.0_f64 - 1.5) / (1.0 + 1.5)).powi(2);
    assert!(approx(r, r_bare, 1e-3));
}

#[test]
fn test_fiber_zero_length() {
    let p = fiber_attenuation(1.0, 0.2, 0.0);
    assert!(approx(p, 1.0, EPS));
}

#[test]
fn test_matrix_mul_associativity() {
    let a = Matrix2x2::propagation(1.0);
    let b = Matrix2x2::thin_lens(0.5);
    let c = Matrix2x2::propagation(2.0);
    let ab_c = a.matmul(b).matmul(c);
    let a_bc = a.matmul(b.matmul(c));
    assert!(approx(ab_c.a, a_bc.a, EPS));
    assert!(approx(ab_c.b, a_bc.b, EPS));
    assert!(approx(ab_c.c, a_bc.c, EPS));
    assert!(approx(ab_c.d, a_bc.d, EPS));
}
