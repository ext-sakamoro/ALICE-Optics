**English** | [日本語](README_JP.md)

# ALICE-Optics

Optical simulation library for the ALICE ecosystem. Covers geometric optics, wave optics, polarization, fiber optics, and aberration analysis -- all in pure Rust with zero dependencies.

## Features

- **Geometric Optics** -- Snell's law (scalar and vector), Fresnel equations, Brewster/critical angles
- **Lens Systems** -- Thin/thick lens equations, lensmaker's equation, ABCD ray transfer matrices, paraxial ray tracing
- **Diffraction** -- Single/double slit intensity, Airy disk, Rayleigh resolution, diffraction gratings
- **Polarization** -- Jones vectors/matrices, Stokes vectors, Mueller matrices, wave plates, polarizers
- **Thin Film Interference** -- Reflectance, anti-reflection coating design, constructive/destructive wavelengths
- **Fiber Optics** -- Numerical aperture, V-number, mode count, attenuation, modal dispersion
- **Aberrations** -- Spherical, chromatic, coma, Petzval curvature, distortion, Abbe number, achromatic doublet
- **Gaussian Beams** -- Rayleigh range, beam waist, divergence, Gouy phase
- **Prisms & Interferometry** -- Minimum deviation, angular dispersion, OPD, coherence length, fringe visibility
- **Caustics** -- Light refracted through a height-field surface and concentrated on a floor, as the reciprocal Jacobian determinant of the landing map; obliquity, Fresnel transmittance and Beer-Lambert attenuation included, plus bilinear deposition of the field onto a floor grid with the spilled power reported

## Architecture

```
Ray / Vec3           Geometric core (Snell, Fresnel, lenses)
    |
Matrix2x2            ABCD ray transfer matrices
    |
JonesVector/Matrix   Polarization (Jones calculus)
StokesVector         Polarization (Mueller calculus)
    |
Diffraction          Wave optics (slits, gratings, Airy)
    |
Fiber / Aberration   Applied optics modules
Caustics             Surface -> floor irradiance (Snell + Fresnel + Beer-Lambert)
```

Every expectation in `tests/analytic_caustics.rs` is a closed form written out by
hand -- an affine surface cannot concentrate light, normal-incidence transmittance
into water is exactly 48/49, a parabolic dip of radius R focuses at R / (1 - n1/n2)
-- never a value the tracer produced.

## License

MIT OR Apache-2.0
