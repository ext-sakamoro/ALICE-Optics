//! Caustics: light refracted through a surface and concentrated on a floor.
//!
//! The forward method. Each node of a height-field lattice sends one ray of light
//! through the interface (Snell) down to a flat floor. The map from the lattice to
//! the landing points is a deformation of the plane; where it compresses, light
//! concentrates. The concentration factor is the reciprocal of the absolute
//! Jacobian determinant of that map, so a patch whose light lands on a smaller
//! patch is brighter by exactly the ratio of the two areas.
//!
//! Nothing here is stochastic and nothing is stored per photon: the field is a
//! deterministic function of the surface, the light direction and the medium.
//!
//! ```
//! use alice_optics::caustics::{trace_caustics, CausticConfig, SurfaceGrid};
//! use alice_optics::vec3::Vec3;
//!
//! // A flat surface cannot focus: every landing patch has the same area as the
//! // lattice patch it came from, so the concentration is 1 everywhere.
//! let heights = vec![0.0; 9];
//! let normals = vec![Vec3::new(0.0, 1.0, 0.0); 9];
//! let grid = SurfaceGrid::new(3, 3, 2.0, 2.0, &heights, &normals).unwrap();
//! let cfg = CausticConfig::preset_air_to_water(-1.0).unwrap();
//! let field = trace_caustics(&grid, Vec3::new(0.0, -1.0, 0.0), &cfg).unwrap();
//! assert!((field.at(1, 1).concentration - 1.0).abs() < 1e-9);
//! ```

// Lattice indices are bounded by the grid the caller allocated, so turning one
// into a coordinate cannot lose anything an f64 would notice.
#![allow(clippy::cast_precision_loss)]

use crate::fresnel::fresnel_unpolarized;
use crate::snell::snell_vec;
use crate::vec3::Vec3;

/// Refractive index of water used by [`CausticConfig::preset_air_to_water`].
///
/// Exactly `4/3` rather than the measured `1.333`, so that the paraxial
/// deflection factor `1 - 1/n` is exactly `0.25` and closed-form expectations
/// stay exact.
pub const IOR_WATER: f64 = 4.0 / 3.0;

/// Refractive index of air.
pub const IOR_AIR: f64 = 1.0;

/// Why a caustic field could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CausticError {
    /// A lattice needs at least two nodes along each axis to have a spacing.
    GridTooSmall,
    /// `heights` or `normals` did not have `nx * nz` entries.
    GridLengthMismatch,
    /// A lattice extent was not finite and positive.
    GridExtentInvalid,
    /// A refractive index was not finite, or was below 1.
    IorInvalid,
    /// The absorption coefficient was not finite, or was negative.
    AbsorptionInvalid,
    /// The source irradiance was not finite, or was not positive.
    IrradianceInvalid,
    /// The floor height was not finite.
    FloorInvalid,
    /// The light direction was degenerate, or did not travel downward.
    LightDirectionInvalid,
    /// A lattice node sits at or below the floor, so no ray can reach it.
    FloorAboveSurface {
        /// Offending lattice index, in `j * nx + i` order.
        index: usize,
    },
    /// The ray at this lattice node was totally internally reflected.
    ///
    /// Cannot happen going from a thinner into a denser medium (air into water),
    /// which is why it is an error rather than a hole in the field.
    TotalInternalReflection {
        /// Offending lattice index, in `j * nx + i` order.
        index: usize,
    },
    /// The refracted ray at this lattice node did not travel downward, so it
    /// never meets the floor.
    RayMissesFloor {
        /// Offending lattice index, in `j * nx + i` order.
        index: usize,
    },
    /// The surface at this lattice node turns its back on the light.
    ///
    /// Refracting anyway would silently treat the ray as arriving from below the
    /// interface and return a plausible-looking number for an unlit node.
    SurfaceFacesAway {
        /// Offending lattice index, in `j * nx + i` order.
        index: usize,
    },
}

/// The medium and the geometry a caustic is computed in.
///
/// Fields are private and every constructor validates, so a configuration that
/// exists is one the tracer can use. There is deliberately no `Default`: a
/// refractive index and a floor height carry physical meaning, and choosing them
/// silently would make every number downstream unattributable.
///
/// Built from outside the crate with [`Self::try_new`] or
/// [`Self::preset_air_to_water`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CausticConfig {
    ior_above: f64,
    ior_below: f64,
    floor_y: f64,
    absorption: f64,
    source_irradiance: f64,
}

impl CausticConfig {
    /// Build a configuration, checking every value up front.
    ///
    /// `absorption` is the Beer-Lambert coefficient per unit of travelled length
    /// below the interface; `source_irradiance` is measured on a plane facing the
    /// source.
    ///
    /// # Errors
    /// [`CausticError::IorInvalid`] if either index is not finite or is below 1,
    /// [`CausticError::AbsorptionInvalid`] if absorption is not finite or negative,
    /// [`CausticError::IrradianceInvalid`] if the irradiance is not finite or not
    /// positive, [`CausticError::FloorInvalid`] if the floor height is not finite.
    pub fn try_new(
        ior_above: f64,
        ior_below: f64,
        floor_y: f64,
        absorption: f64,
        source_irradiance: f64,
    ) -> Result<Self, CausticError> {
        if !ior_above.is_finite() || !ior_below.is_finite() || ior_above < 1.0 || ior_below < 1.0 {
            return Err(CausticError::IorInvalid);
        }
        if !absorption.is_finite() || absorption < 0.0 {
            return Err(CausticError::AbsorptionInvalid);
        }
        if !source_irradiance.is_finite() || source_irradiance <= 0.0 {
            return Err(CausticError::IrradianceInvalid);
        }
        if !floor_y.is_finite() {
            return Err(CausticError::FloorInvalid);
        }
        Ok(Self {
            ior_above,
            ior_below,
            floor_y,
            absorption,
            source_irradiance,
        })
    }

    /// Sunlight entering clear water, with the floor at `floor_y`.
    ///
    /// Absorption is zero, which is the right model over pool depths and keeps the
    /// energy balance exact; use [`Self::try_new`] or [`Self::with_absorption`]
    /// for water deep enough that colour matters.
    ///
    /// # Errors
    /// [`CausticError::FloorInvalid`] if `floor_y` is not finite.
    pub fn preset_air_to_water(floor_y: f64) -> Result<Self, CausticError> {
        Self::try_new(IOR_AIR, IOR_WATER, floor_y, 0.0, 1.0)
    }

    /// Refractive index above the interface.
    #[must_use]
    pub const fn ior_above(self) -> f64 {
        self.ior_above
    }

    /// Refractive index below the interface.
    #[must_use]
    pub const fn ior_below(self) -> f64 {
        self.ior_below
    }

    /// Height of the floor plane.
    #[must_use]
    pub const fn floor_y(self) -> f64 {
        self.floor_y
    }

    /// Beer-Lambert absorption coefficient per unit of travelled length.
    #[must_use]
    pub const fn absorption(self) -> f64 {
        self.absorption
    }

    /// Irradiance of the source, measured on a plane facing it.
    #[must_use]
    pub const fn source_irradiance(self) -> f64 {
        self.source_irradiance
    }

    /// The same configuration with a different absorption coefficient.
    ///
    /// # Errors
    /// [`CausticError::AbsorptionInvalid`] if not finite or negative.
    pub fn with_absorption(self, absorption: f64) -> Result<Self, CausticError> {
        Self::try_new(
            self.ior_above,
            self.ior_below,
            self.floor_y,
            absorption,
            self.source_irradiance,
        )
    }
}

/// A height field sampled on a regular lattice in the xz plane, centred on the origin.
///
/// `heights[j * nx + i]` is the displacement of the surface above `y = 0` at node
/// `(i, j)`, and `normals[j * nx + i]` is the upward surface normal there. Normals
/// are supplied rather than derived so that a caller holding an analytic surface
/// can pass exact ones instead of a difference quotient.
///
/// Built from outside the crate with [`Self::new`].
#[derive(Debug, Clone, Copy)]
pub struct SurfaceGrid<'a> {
    nx: usize,
    nz: usize,
    extent_x: f64,
    extent_z: f64,
    heights: &'a [f64],
    normals: &'a [Vec3],
}

impl<'a> SurfaceGrid<'a> {
    /// Build a lattice, checking the shape up front.
    ///
    /// # Errors
    /// [`CausticError::GridTooSmall`] below 2x2, [`CausticError::GridLengthMismatch`]
    /// if the slices are not `nx * nz` long, [`CausticError::GridExtentInvalid`] if
    /// an extent is not finite and positive.
    pub fn new(
        nx: usize,
        nz: usize,
        extent_x: f64,
        extent_z: f64,
        heights: &'a [f64],
        normals: &'a [Vec3],
    ) -> Result<Self, CausticError> {
        if nx < 2 || nz < 2 {
            return Err(CausticError::GridTooSmall);
        }
        let expected = nx.checked_mul(nz).ok_or(CausticError::GridLengthMismatch)?;
        if heights.len() != expected || normals.len() != expected {
            return Err(CausticError::GridLengthMismatch);
        }
        if !extent_x.is_finite() || !extent_z.is_finite() || extent_x <= 0.0 || extent_z <= 0.0 {
            return Err(CausticError::GridExtentInvalid);
        }
        Ok(Self {
            nx,
            nz,
            extent_x,
            extent_z,
            heights,
            normals,
        })
    }

    /// Number of nodes along x.
    #[must_use]
    pub const fn nx(self) -> usize {
        self.nx
    }

    /// Number of nodes along z.
    #[must_use]
    pub const fn nz(self) -> usize {
        self.nz
    }

    /// Distance between neighbouring nodes along x.
    #[must_use]
    pub fn spacing_x(self) -> f64 {
        self.extent_x / (self.nx - 1) as f64
    }

    /// Distance between neighbouring nodes along z.
    #[must_use]
    pub fn spacing_z(self) -> f64 {
        self.extent_z / (self.nz - 1) as f64
    }

    /// World position of lattice node `(i, j)`.
    ///
    /// # Panics
    /// If `i >= nx` or `j >= nz`.
    #[must_use]
    pub fn position(self, i: usize, j: usize) -> Vec3 {
        assert!(i < self.nx && j < self.nz, "lattice index out of range");
        let x = (i as f64).mul_add(self.spacing_x(), -self.extent_x / 2.0);
        let z = (j as f64).mul_add(self.spacing_z(), -self.extent_z / 2.0);
        Vec3::new(x, self.heights[j * self.nx + i], z)
    }

    /// Upward surface normal at lattice node `(i, j)`.
    ///
    /// # Panics
    /// If `i >= nx` or `j >= nz`.
    #[must_use]
    pub fn normal(self, i: usize, j: usize) -> Vec3 {
        assert!(i < self.nx && j < self.nz, "lattice index out of range");
        self.normals[j * self.nx + i]
    }
}

/// One lattice node's contribution to the floor.
///
/// Produced by [`trace_caustics`] and read by the caller; it is not built outside
/// this crate, which is why it has no public constructor.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CausticSample {
    /// Where this node's ray lands on the floor.
    pub landing: Vec3,
    /// Irradiance leaving the interface, per unit of lattice footprint area.
    ///
    /// Obliquity, the transmitted Fresnel fraction and Beer-Lambert attenuation
    /// along the path to the floor are already folded in.
    pub transmitted: f64,
    /// Lattice footprint area divided by the area it lands on.
    ///
    /// `1.0` where the surface does not bend the beam. Diverges at a focus, where
    /// the landing area goes to zero; the value is then [`f64::INFINITY`] rather
    /// than a clamped stand-in, so that a caller sees the singularity instead of a
    /// plausible-looking number.
    pub concentration: f64,
}

impl CausticSample {
    /// Irradiance actually received by the floor at [`Self::landing`].
    #[must_use]
    pub fn floor_irradiance(self) -> f64 {
        self.transmitted * self.concentration
    }
}

/// A caustic field sampled on the same lattice as the surface that made it.
///
/// Produced by [`trace_caustics`] and read by the caller; it is not built outside
/// this crate, which is why it has no public constructor.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CausticField {
    /// Nodes along x.
    pub nx: usize,
    /// Nodes along z.
    pub nz: usize,
    /// Footprint area of one lattice cell, measured in the xz plane.
    pub cell_area: f64,
    /// One entry per lattice node, in `j * nx + i` order.
    pub samples: Vec<CausticSample>,
}

impl CausticField {
    /// Sample at lattice node `(i, j)`.
    ///
    /// # Panics
    /// If `i >= nx` or `j >= nz`.
    #[must_use]
    pub fn at(&self, i: usize, j: usize) -> CausticSample {
        assert!(i < self.nx && j < self.nz, "lattice index out of range");
        self.samples[j * self.nx + i]
    }
}

/// Where one node's ray lands and what fraction of the source it still carries.
struct NodeTrace {
    landing: Vec3,
    /// Everything except the obliquity factor, which is shared by the whole field.
    carried: f64,
}

/// Send one ray through the interface and down to the floor.
fn trace_node(
    position: Vec3,
    normal: Vec3,
    dir: Vec3,
    cfg: &CausticConfig,
    index: usize,
) -> Result<NodeTrace, CausticError> {
    if position.y <= cfg.floor_y {
        return Err(CausticError::FloorAboveSurface { index });
    }
    let n = normal.normalized();
    let cos_i = -dir.dot(n);
    if cos_i <= 0.0 {
        return Err(CausticError::SurfaceFacesAway { index });
    }
    let refracted = snell_vec(dir, n, cfg.ior_above, cfg.ior_below)
        .ok_or(CausticError::TotalInternalReflection { index })?;
    if refracted.y >= 0.0 {
        return Err(CausticError::RayMissesFloor { index });
    }

    let path = (position.y - cfg.floor_y) / -refracted.y;
    let landing = position.plus(refracted.scale(path));
    // `cos_i` can overshoot 1 by a rounding step; `acos` would give NaN there.
    let theta_i = cos_i.min(1.0).acos();
    let transmittance = 1.0 - fresnel_unpolarized(cfg.ior_above, theta_i, cfg.ior_below);
    let carried = cfg.source_irradiance * transmittance * (-cfg.absorption * path).exp();

    Ok(NodeTrace { landing, carried })
}

/// The two nodes a derivative along one axis is taken between, and their distance.
///
/// Central inside, one-sided at the two borders, which is why a caller comparing
/// against a geometric area should stay off the border ring.
const fn stencil(index: usize, count: usize, step: f64) -> (usize, usize, f64) {
    if index == 0 {
        (0, 1, step)
    } else if index == count - 1 {
        (count - 2, count - 1, step)
    } else {
        (index - 1, index + 1, 2.0 * step)
    }
}

/// Refract the light at every lattice node and measure where it concentrates.
///
/// `light_dir` is the direction the light *travels*, so it points downward.
///
/// # Errors
/// Any variant of [`CausticError`]; see its documentation.
pub fn trace_caustics(
    grid: &SurfaceGrid<'_>,
    light_dir: Vec3,
    cfg: &CausticConfig,
) -> Result<CausticField, CausticError> {
    if !light_dir.x.is_finite() || !light_dir.y.is_finite() || !light_dir.z.is_finite() {
        return Err(CausticError::LightDirectionInvalid);
    }
    let dir = light_dir.normalized();
    // A degenerate direction normalizes to zero, whose y is not below zero either.
    if dir.y >= 0.0 {
        return Err(CausticError::LightDirectionInvalid);
    }

    let nx = grid.nx();
    let nz = grid.nz();
    // Flux through a horizontal footprint, so the tilt of the beam counts and the
    // tilt of the surface does not; the surface tilt is already in the Fresnel term.
    let obliquity = -dir.y;

    let mut landings = Vec::with_capacity(nx * nz);
    let mut carried = Vec::with_capacity(nx * nz);
    for j in 0..nz {
        for i in 0..nx {
            let index = j * nx + i;
            let node = trace_node(grid.position(i, j), grid.normal(i, j), dir, cfg, index)?;
            landings.push(node.landing);
            carried.push(obliquity * node.carried);
        }
    }

    let dx = grid.spacing_x();
    let dz = grid.spacing_z();
    let mut samples = Vec::with_capacity(nx * nz);
    for j in 0..nz {
        for i in 0..nx {
            let (lo_i, hi_i, span_x) = stencil(i, nx, dx);
            let (lo_j, hi_j, span_z) = stencil(j, nz, dz);
            let d_dx = landings[j * nx + hi_i]
                .minus(landings[j * nx + lo_i])
                .scale(1.0 / span_x);
            let d_dz = landings[hi_j * nx + i]
                .minus(landings[lo_j * nx + i])
                .scale(1.0 / span_z);
            // Jacobian determinant of the landing map, taken in the xz plane.
            let det = d_dx.x.mul_add(d_dz.z, -(d_dx.z * d_dz.x));
            // A vanishing determinant is a focus; dividing gives infinity, which is
            // the honest answer, so it is not special-cased into a finite stand-in.
            let concentration = 1.0 / det.abs();
            samples.push(CausticSample {
                landing: landings[j * nx + i],
                transmitted: carried[j * nx + i],
                concentration,
            });
        }
    }

    Ok(CausticField {
        nx,
        nz,
        cell_area: dx * dz,
        samples,
    })
}
