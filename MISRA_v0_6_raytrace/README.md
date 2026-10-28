# MISRA v0.6 — Michelson Interferometry Simulator by Ali

## Scope
A **simulation-only** desktop program. No lens/component editor, phase retrieval, unwrapping or surface reconstruction. Test mirror can be flat, spherical or parabolic, with signed radius and tilts. Every detector pixel has a reverse-traced ray in **each arm**, with explicit plane beam splitter, mirror intersections, reflection normals, return to beam splitter, and continuation to the input wavefront plane. Rays failing the aperture or intersection validity checks are masked.

## Installation
Python 3.10+ required. On Windows, run `start_windows.bat`; alternatively:

```bash
python -m pip install -r requirements.txt
python misra.py
```

Headless engine tests: `python -m pytest tests` (requires `pip install pytest`).

## Coordinates and optical model
- Beam splitter: ideal zero-thickness 45-degree plane **x=y**.
- Source input: x=-S; reference mirror: y=Lr; test mirror: x=Lt+sag(y,z); detector: y=-D.
- Detector rays are orthographic and traced **backwards** through the arms. Reflection uses the exact local surface normal and intersection with the test surface is solved iteratively with Newton's method.
- Reflection/transmission at the ideal splitter changes direction geometrically; splitter phase and polarization are simplified to equal amplitude, a fixed relative phase convention, and no wavelength-dependent coating.
- Plane source: constant phase at source plane. Point source: spherical eikonal phase evaluated at source plane, with virtual source x=-2S. This is an *approximate wavefront boundary condition*, not full point-source diffraction.
- Optical path = sum of traced Euclidean segment lengths plus optional source wavefront phase path. Relative phase is 2*pi*(OPL_test-OPL_ref)/wavelength.
- Detector normalized intensity = 0.5 + 0.45*cos(phase + phase_step), with optional additive Gaussian noise and clipping. No photon statistics, sensor MTF or coherent field propagation.
- Ray paths displayed in 3D are **backward** ray paths, not a full bidirectional field solution.

## Important physical limitations
This is **geometric-optics / eikonal** ray tracing, not a full-wave electromagnetic model or a validated digital twin. It traces one ray per pixel per arm; **not every possible light path**, and may miss caustics, multiple ray solutions, vignetting at BS, and non-paraxial coherent amplitude variations. Backward rays are used to sample an input wavefront, so this is not exact ray shooting from a finite point emitter. BS is idealized; there is no glass thickness, Fresnel polarization, phase shift from real coatings, diffraction, speckle, multiple reflections, detector PSF or physically calibrated absolute intensity. Do not interpret this output as indistinguishable from laboratory data.

## Data export
`detector_dataset.npz` contains `intensity` [frame,row,col], `x_m`, `z_m`, `phase_shifts_rad`, `valid_mask`; `detector_00.csv` etc, 16-bit PNGs, and `metadata.json`. Detector export **does not include true surface or OPD**. Separate `SIMULATION_TRUTH_DO_NOT_FEED_RECONSTRUCTION.npz` contains truth for future blind reconstruction validation; never feed it into the complementary reconstruction program.

## Validating the simulation
Test known flat, positive and negative curvature, change tilt, check fringe motion and compare phase shifts. For high fidelity, next steps are coherent complex-field propagation, physical BS Fresnel matrices and calibration with measured hardware.
