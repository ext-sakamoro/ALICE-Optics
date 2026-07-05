//! Optical ray.

use crate::vec3::Vec3;

/// An optical ray with origin, direction, and wavelength (in metres).
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub wavelength: f64,
}

impl Ray {
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3, wavelength: f64) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
            wavelength,
        }
    }

    /// Propagate the ray by distance `t`.
    #[must_use]
    pub fn at(self, t: f64) -> Vec3 {
        self.origin.plus(self.direction.scale(t))
    }
}
