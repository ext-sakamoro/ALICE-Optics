//! Convenience re-export (= `use alice_optics::prelude::*;`).

pub use crate::aberrations::{
    abbe_number, achromatic_doublet, chromatic_aberration, coma_coefficient, distortion_percent,
    petzval_radius, spherical_aberration_longitudinal, spherical_aberration_transverse,
};
pub use crate::diffraction::{
    airy_first_zero, double_slit_intensity, double_slit_maxima, grating_maxima,
    rayleigh_resolution, single_slit_intensity, single_slit_minima,
};
pub use crate::dispersion::{cauchy_dispersion, group_index, sellmeier_single};
pub use crate::fiber::{
    fiber_acceptance_angle, fiber_attenuation, fiber_is_single_mode, fiber_na, fiber_num_modes,
    fiber_v_number, modal_dispersion,
};
pub use crate::fresnel::{
    brewster_angle, critical_angle, fresnel_rp, fresnel_rs, fresnel_unpolarized,
};
pub use crate::gaussian_beam::{
    beam_divergence, beam_waist_at_z, gouy_phase, rayleigh_range, wavefront_radius,
};
pub use crate::interference::{
    coherence_length, fringe_visibility, optical_path_length, phase_from_opd, two_beam_interference,
};
pub use crate::jones::{JonesMatrix, JonesVector};
pub use crate::matrix_optics::{compose_matrices, trace_paraxial, Matrix2x2, SphericalSurface};
pub use crate::prism::{prism_angular_dispersion, prism_index_from_deviation, prism_min_deviation};
pub use crate::ray::Ray;
pub use crate::snell::{snell, snell_vec};
pub use crate::stokes::{MuellerMatrix, StokesVector};
pub use crate::thick_lens::{
    thick_lens_back_principal, thick_lens_focal_length, thick_lens_front_principal,
};
pub use crate::thin_film::{
    anti_reflection_index, anti_reflection_thickness, thin_film_constructive_lambda,
    thin_film_destructive_lambda, thin_film_reflectance,
};
pub use crate::thin_lens::{
    lensmaker_thin, optical_power, thin_lens_image_distance, thin_lens_magnification,
};
pub use crate::vec3::Vec3;
