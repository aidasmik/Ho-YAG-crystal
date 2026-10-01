"""Per-encounter geometry of a mirror-array thin-disk multipass amplifier.

In a mirror-array multipass the seed fans between the disk and an array of
curved mirrors. Every disk encounter therefore has its own angle of incidence,
every return path its own length and mirror, and every mirror its own small
pointing error. This module describes that geometry and applies it to a scalar
field on the transverse optical grid (beam coordinates):

* At encounter k the beam sees the disk-plane OPD map stretched by
  1/cos(theta_k) along the incidence plane and scaled by 1/cos(theta_t) for the
  longer internal path (theta_t from Snell's law).
* Between encounters the field propagates L_k/2 to its array mirror, receives
  the mirror's focusing (astigmatic at oblique incidence: f cos(a) tangential,
  f/cos(a) sagittal) and pointing error (2 epsilon deflection), then
  propagates L_k/2 back to the disk. Pointing errors walk the beam on the disk.

Image-relay layout (default): every return path is a unit-magnification 4f
relay (for example a parabolic mirror with fold prisms) that images the disk
back onto itself. The field arrives point-inverted, x -> -x and y -> -y, with no
net free-space diffraction. What differs per pass is the angle of incidence on
the disk, a lateral image shift 2*epsilon*f from each fold mirror's pointing
error, and a small path-length (focus) error that propagates the image by
defocus_m.

Bounces: an encounter is one traversal of the crystal. A thin-disk bounce is
two traversals (in, reflection at the HR-coated back, out) at one angle of
incidence with no relay between them; relays connect successive bounces only.

Not modelled: the elongated gain footprint on the disk (the population is
sampled in beam coordinates), image inversion by the mirror-array sequence,
relay aberrations and apertures, and polarization. With angles below ~10
degrees the footprint change is below 1.5%.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from hoyag.propagation import angular_spectrum_propagate

YAG_INDEX_1030 = 1.815


@dataclass(frozen=True)
class RelayPath:
    length_m: float
    focal_m: float = 0.            # zero: no focusing mirror
    mirror_incidence_rad: float = 0.
    tilt_rad: tuple = (0., 0.)     # mirror pointing error about y and x
    kind: str = 'free_space'       # 'free_space' or 'image' (4f relay)
    defocus_m: float = 0.          # image relay: path-length (focus) error

    def __post_init__(self):
        if (self.kind not in ('free_space', 'image', 'reflection') or
                not math.isfinite(self.defocus_m) or
                not math.isfinite(self.length_m) or self.length_m <= 0 or
                not math.isfinite(self.focal_m) or
                not 0 <= self.mirror_incidence_rad < math.pi/3 or
                len(self.tilt_rad) != 2 or not all(math.isfinite(t) for t in self.tilt_rad)):
            raise ValueError('invalid multipass relay path')


@dataclass(frozen=True)
class MultipassGeometry:
    incidence_rad: tuple           # angle of incidence on the disk, per encounter
    paths: tuple                   # RelayPath from encounter k to k+1
    plane_rad: float = 0.          # orientation of the incidence plane (0 = x)
    index: float = YAG_INDEX_1030
    layout: str = 'custom'

    def __post_init__(self):
        if (len(self.incidence_rad) < 1 or len(self.paths) != len(self.incidence_rad)-1 or
                not all(0 <= abs(t) < math.pi/3 for t in self.incidence_rad) or
                not all(isinstance(p, RelayPath) for p in self.paths) or
                not math.isfinite(self.plane_rad) or not self.index >= 1):
            raise ValueError('invalid multipass geometry')

    @property
    def encounters(self):
        return len(self.incidence_rad)

    def _beam_axes(self, grid):
        x, y = grid.mesh
        c, s = math.cos(self.plane_rad), math.sin(self.plane_rad)
        return x, y, c*x+s*y, -s*x+c*y, c, s

    def encounter_map(self, k, disk_map, grid):
        """Disk-plane OPD (or phase) map as seen by the beam at encounter k."""
        disk_map = np.asarray(disk_map, float)
        theta = float(self.incidence_rad[k])
        internal = math.asin(math.sin(abs(theta))/self.index)
        if theta == 0:
            return disk_map/math.cos(internal)
        rows, cols = self.sampling(k, grid)
        sampled = map_coordinates(disk_map, [rows, cols], order=1, mode='nearest')
        return sampled/math.cos(internal)

    def sampling(self, k, grid):
        """Fractional disk-map (row, col) indices seen at encounter k."""
        _, _, u, v, c, s = self._beam_axes(grid)
        u = u/math.cos(float(self.incidence_rad[k]))
        xd, yd = c*u-s*v, s*u+c*v
        return (yd-grid.y[0])/grid.dy, (xd-grid.x[0])/grid.dx

    def internal_scale(self, k):
        return 1/math.cos(math.asin(math.sin(abs(float(self.incidence_rad[k])))/self.index))

    def mirror_phase(self, k, grid, wavelength_m, *, include_tilt=True):
        path = self.paths[k]
        if path.kind in ('image', 'reflection'):
            return np.ones(grid.shape, complex)
        x, y, u, v, _, _ = self._beam_axes(grid)
        wave = 2*math.pi/wavelength_m
        phase = np.zeros(grid.shape)
        if path.focal_m:
            a = path.mirror_incidence_rad
            phase -= wave/2*(u*u/(path.focal_m*math.cos(a))+v*v*math.cos(a)/path.focal_m)
        if include_tilt:
            phase += wave*2*(path.tilt_rad[0]*x+path.tilt_rad[1]*y)
        return np.exp(1j*phase)

    def mirror_phases(self, grid, wavelength_m):
        return [self.mirror_phase(k, grid, wavelength_m) for k in range(len(self.paths))]

    def image_shift_m(self, k):
        """Lateral image shift on the disk from the fold-mirror pointing error."""
        path = self.paths[k]
        if path.kind != 'image':
            return 0., 0.
        return 2*path.focal_m*path.tilt_rad[0], 2*path.focal_m*path.tilt_rad[1]

    def _image(self, k, field, grid, wavelength_m, sign):
        path = self.paths[k]
        field = np.asarray(field, complex)
        dx, dy = self.image_shift_m(k)
        if sign > 0:
            field = field[::-1, ::-1]
        if dx or dy:
            fx, fy = np.meshgrid(grid.fx, grid.fy, indexing='xy')
            ramp = np.exp(-2j*np.pi*sign*(fx*dx+fy*dy))
            field = np.fft.ifft2(np.fft.fft2(field)*ramp)
        if path.defocus_m:
            field = angular_spectrum_propagate(field, grid, wavelength_m, sign*path.defocus_m)
        if sign < 0:
            field = field[::-1, ::-1]
        return field

    def relay(self, k, field, grid, wavelength_m, mirror=None):
        if self.paths[k].kind == 'reflection':
            return np.asarray(field, complex)
        if self.paths[k].kind == 'image':
            return self._image(k, field, grid, wavelength_m, 1)
        half = self.paths[k].length_m/2
        mirror = self.mirror_phase(k, grid, wavelength_m) if mirror is None else mirror
        field = angular_spectrum_propagate(field, grid, wavelength_m, half)*mirror
        return angular_spectrum_propagate(field, grid, wavelength_m, half)

    def relay_adjoint(self, k, field, grid, wavelength_m, mirror=None):
        if self.paths[k].kind == 'reflection':
            return np.asarray(field, complex)
        if self.paths[k].kind == 'image':
            return self._image(k, field, grid, wavelength_m, -1)
        half = self.paths[k].length_m/2
        mirror = self.mirror_phase(k, grid, wavelength_m) if mirror is None else mirror
        field = angular_spectrum_propagate(field, grid, wavelength_m, -half)*np.conj(mirror)
        return angular_spectrum_propagate(field, grid, wavelength_m, -half)

    def summary(self):
        return {'layout': self.layout,
                'disk_bounces': 1+sum(p.kind != HR_REFLECTION for p in self.paths),
                'relay_kind': [p.kind for p in self.paths],
                'relay_defocus_mm': [1e3*p.defocus_m for p in self.paths],
                'image_shift_um': [[1e6*s for s in self.image_shift_m(k)]
                                   for k in range(len(self.paths))],
                'incidence_deg': [math.degrees(t) for t in self.incidence_rad],
                'path_length_m': [p.length_m for p in self.paths],
                'mirror_focal_m': [p.focal_m for p in self.paths],
                'mirror_incidence_deg': [math.degrees(p.mirror_incidence_rad) for p in self.paths],
                'mirror_tilt_urad': [[1e6*t for t in p.tilt_rad] for p in self.paths],
                'incidence_plane_deg': math.degrees(self.plane_rad), 'index': self.index}


def mode_matched_focal(length_m, waist_m, wavelength_m):
    """Mirror focal length at mid-path that reproduces a Gaussian waist on the disk."""
    rayleigh = math.pi*waist_m**2/wavelength_m
    return (length_m**2/4+rayleigh**2)/length_m


HR_REFLECTION = 'reflection'


def _bounces(encounters, per_bounce):
    """Bounce index of every traversal and whether gap k is inside a bounce."""
    if per_bounce not in (1, 2) or encounters < 1:
        raise ValueError('traversals per bounce must be 1 or 2')
    bounce = [k//per_bounce for k in range(encounters)]
    inside = [bounce[k] == bounce[k+1] for k in range(encounters-1)]
    return bounce, inside, bounce[-1]+1


def _reflection():
    return RelayPath(1e-9, kind=HR_REFLECTION)


def uniform(encounters, distance_m, focal_m=0., per_bounce=2):
    _, inside, _ = _bounces(encounters, per_bounce)
    return MultipassGeometry(tuple([0.]*encounters),
        tuple(_reflection() if within else RelayPath(distance_m, focal_m) for within in inside),
        layout='uniform')


def mirror_array(encounters, *, array_distance_m, array_half_height_m, waist_m,
                 wavelength_m, mirror_incidence_deg=1.5, mirror_focal_m=None,
                 tilt_rad=None, incidence_plane_deg=0., index=YAG_INDEX_1030, per_bounce=2):
    """Fan geometry of the diagram: the beam sweeps the array from top to bottom.

    Bounce b meets the disk from array height y_b (linearly spaced over
    +-array_half_height_m); both traversals of a bounce share its angle. The
    path from bounce b to b+1 returns via the mirror at the mean height of y_b
    and y_{b+1}: length 2*sqrt(D^2 + y^2), so outer paths are longer.
    ``tilt_rad`` has one entry per traversal gap; gaps inside a bounce ignore it.
    """
    if encounters < 1 or array_distance_m <= 0 or array_half_height_m < 0:
        raise ValueError('invalid mirror-array geometry')
    bounce, inside, count = _bounces(encounters, per_bounce)
    heights = np.linspace(array_half_height_m, -array_half_height_m, count)
    incidence = tuple(float(math.atan(abs(heights[b])/array_distance_m)) for b in bounce)
    tilt = np.zeros((encounters-1, 2)) if tilt_rad is None else np.asarray(tilt_rad, float)
    if tilt.shape != (encounters-1, 2):
        raise ValueError('one mirror pointing error pair per traversal gap is required')
    paths = []
    for k in range(encounters-1):
        if inside[k]:
            paths.append(_reflection())
            continue
        b = bounce[k]
        length = 2*math.hypot(array_distance_m, .5*(heights[b]+heights[b+1]))
        focal = (mode_matched_focal(length, waist_m, wavelength_m) if mirror_focal_m is None
                 else float(mirror_focal_m))
        paths.append(RelayPath(length, focal, math.radians(mirror_incidence_deg),
                               tuple(float(t) for t in tilt[k])))
    return MultipassGeometry(incidence, tuple(paths), math.radians(incidence_plane_deg),
                             index, layout='mirror_array')


def image_relay(encounters, *, relay_focal_m, max_incidence_deg=8., tilt_rad=None,
                defocus_m=None, incidence_plane_deg=0., index=YAG_INDEX_1030, per_bounce=2):
    """Unit-magnification 4f re-imaging multipass (e.g. parabolic mirror + prisms).

    Bounce b meets the disk at an angle linearly spaced from +max_incidence_deg
    to -max_incidence_deg; both traversals of a bounce share it. Each relay
    between bounces has path length 4f, a fold-mirror pointing error (image
    shift 2*epsilon*f) and a focus error. ``tilt_rad``/``defocus_m`` have one
    entry per traversal gap; gaps inside a bounce ignore them.
    """
    if encounters < 1 or relay_focal_m <= 0 or not 0 <= max_incidence_deg < 60:
        raise ValueError('invalid image-relay geometry')
    bounce, inside, count = _bounces(encounters, per_bounce)
    per = np.radians(np.linspace(max_incidence_deg, -max_incidence_deg, count))
    angles = [per[b] for b in bounce]
    tilt = np.zeros((encounters-1, 2)) if tilt_rad is None else np.asarray(tilt_rad, float)
    focus = np.zeros(encounters-1) if defocus_m is None else np.asarray(defocus_m, float)
    if tilt.shape != (encounters-1, 2) or focus.shape != (encounters-1,):
        raise ValueError('one pointing error pair and focus error per relay is required')
    paths = tuple(_reflection() if inside[k] else
                  RelayPath(4*relay_focal_m, relay_focal_m, 0., tuple(float(t) for t in tilt[k]),
                            kind='image', defocus_m=float(focus[k]))
                  for k in range(encounters-1))
    return MultipassGeometry(tuple(float(a) for a in angles), paths,
                             math.radians(incidence_plane_deg), index, layout='image_relay')


def sample_relay_defocus(encounters, sigma_mm, seed):
    """Fixed per-setup path-length (focus) error of every image relay (metres)."""
    return np.random.default_rng(seed).normal(0, sigma_mm*1e-3, max(encounters-1, 0))


def sample_mirror_tilts(encounters, sigma_urad, seed):
    """Fixed per-setup pointing error of every array mirror (radians)."""
    return np.random.default_rng(seed).normal(0, sigma_urad*1e-6, (max(encounters-1, 0), 2))


def from_nominal(nominal, encounters, *, waist_m, wavelength_m=1030e-9, tilt_rad=None,
                 defocus_m=None):
    """Geometry from a dataset ``nominal`` block, or None for the ideal relay."""
    layout = nominal.get('multipass')
    if layout and layout.get('layout') == 'image_relay':
        return image_relay(encounters, relay_focal_m=float(layout['relay_focal_m']),
            max_incidence_deg=float(layout.get('max_incidence_deg', 8.)), tilt_rad=tilt_rad,
            defocus_m=defocus_m, incidence_plane_deg=float(layout.get('incidence_plane_deg', 0.)))
    if layout:
        if layout.get('layout', 'mirror_array') != 'mirror_array':
            raise ValueError('unknown multipass layout')
        return mirror_array(encounters, array_distance_m=float(layout['array_distance_m']),
            array_half_height_m=float(layout['array_half_height_m']), waist_m=waist_m,
            wavelength_m=wavelength_m,
            mirror_incidence_deg=float(layout.get('array_mirror_incidence_deg', 1.5)),
            mirror_focal_m=layout.get('mirror_focal_m'), tilt_rad=tilt_rad,
            incidence_plane_deg=float(layout.get('incidence_plane_deg', 0.)))
    distance = float(nominal.get('inter_pass_distance_m', 0.))
    if distance > 0:
        return uniform(encounters, distance, float(nominal.get('inter_pass_focal_m', 0.)))
    return None


DATASET_CONFIG = Path(__file__).resolve().parents[2] / 'config' / 'ybyag_nn_dataset.json'


def dataset_defaults(path=DATASET_CONFIG):
    """Mirror-array layout and mirror-error size of the dataset generator.

    The desktop calculation and the closed-loop controller read these values so
    all three use one geometry definition.
    """
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    layout = config['nominal'].get('multipass') or {}
    return {'multipass_layout': layout.get('layout', 'mirror_array') if layout else 'ideal_relay',
            'signal_traversals': int(config['nominal']['signal_traversals']),
            'relay_focal_m': float(layout.get('relay_focal_m', .5)),
            'max_incidence_deg': float(layout.get('max_incidence_deg', 8.)),
            'relay_defocus_error_mm': float(config['ranges'].get('relay_defocus_error_mm', 0.)),
            'array_distance_m': float(layout.get('array_distance_m', .5)),
            'array_half_height_m': float(layout.get('array_half_height_m', .08)),
            'array_mirror_incidence_deg': float(layout.get('array_mirror_incidence_deg', 1.5)),
            'mirror_focal_m': layout.get('mirror_focal_m'),
            'incidence_plane_deg': float(layout.get('incidence_plane_deg', 0.)),
            'mirror_tilt_error_urad': float(config['ranges'].get('mirror_tilt_error_urad', 0.))}


def from_request(values, encounters, *, waist_m, tilt_seed, wavelength_m=1030e-9,
                 with_errors=True):
    """Geometry from flat desktop/controller values; None for the ideal relay.

    ``with_errors=False`` gives the same array with perfect mirrors, used for
    ideal reference solves.
    """
    values = {**dataset_defaults(), **{k: v for k, v in values.items() if v is not None}}
    layout = values['multipass_layout']
    if layout == 'ideal_relay':
        return None
    if layout not in ('mirror_array', 'image_relay'):
        raise ValueError('multipass_layout must be image_relay, mirror_array or ideal_relay')
    sigma = float(values['mirror_tilt_error_urad'])
    focus_sigma = float(values['relay_defocus_error_mm'])
    if not all(math.isfinite(v) and v >= 0 for v in (sigma, focus_sigma)):
        raise ValueError('mirror tilt and relay focus errors must be finite and nonnegative')
    tilts = sample_mirror_tilts(encounters, sigma, int(tilt_seed)) if with_errors else None
    if layout == 'image_relay':
        focus = (sample_relay_defocus(encounters, focus_sigma, int(tilt_seed)+1)
                 if with_errors else None)
        return image_relay(encounters, relay_focal_m=float(values['relay_focal_m']),
            max_incidence_deg=float(values['max_incidence_deg']), tilt_rad=tilts,
            defocus_m=focus, incidence_plane_deg=float(values['incidence_plane_deg']))
    focal = values.get('mirror_focal_m')
    return mirror_array(encounters, array_distance_m=float(values['array_distance_m']),
        array_half_height_m=float(values['array_half_height_m']), waist_m=waist_m,
        wavelength_m=wavelength_m,
        mirror_incidence_deg=float(values['array_mirror_incidence_deg']),
        mirror_focal_m=None if focal in (None, '') else float(focal), tilt_rad=tilts,
        incidence_plane_deg=float(values['incidence_plane_deg']))
