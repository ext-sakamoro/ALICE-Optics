#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]

//! ALICE-Optics: Optical simulation library.
//!
//! Provides lens systems, ray tracing, diffraction, polarization,
//! thin film interference, Snell/Fresnel equations, fiber optics, and aberrations.

pub mod aberrations;
pub mod diffraction;
pub mod dispersion;
pub mod fiber;
pub mod fresnel;
pub mod gaussian_beam;
pub mod interference;
pub mod jones;
pub mod matrix_optics;
pub mod prelude;
pub mod prism;
pub mod ray;
pub mod snell;
pub mod stokes;
pub mod thick_lens;
pub mod thin_film;
pub mod thin_lens;
pub mod vec3;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::aberrations::*;
pub use crate::diffraction::*;
pub use crate::dispersion::*;
pub use crate::fiber::*;
pub use crate::fresnel::*;
pub use crate::gaussian_beam::*;
pub use crate::interference::*;
pub use crate::jones::*;
pub use crate::matrix_optics::*;
pub use crate::prism::*;
pub use crate::ray::*;
pub use crate::snell::*;
pub use crate::stokes::*;
pub use crate::thick_lens::*;
pub use crate::thin_film::*;
pub use crate::thin_lens::*;
pub use crate::vec3::*;
