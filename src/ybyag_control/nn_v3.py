"""One-shot, measurement-certified Yb:YAG SLM corrector (controller V3).

V2 regresses 14 modal coefficients from a few hundred solver-verified teacher
labels and then scales the proposal. V3 instead makes one full correction per
measured trial and only applies it when the measured frames support it:

1. A network trained on unlimited synthetic passive-optics states (no teacher
   labels) proposes a full 14-mode + dense residual SLM step and an estimate of
   the hidden optical state. Its training loss is the corrected coherent
   fidelity itself, evaluated through the differentiable optical model.
2. The state is refined against both measured phase-diverse camera frames
   (maximum a posteriori phase diversity). A Gauss-Newton Laplace posterior and
   any competing minima become the state uncertainty.
3. The step is re-planned to maximise mean fidelity over posterior samples.
4. Hold and every candidate are scored against each posterior sample with the
   exact quantized SLM rendering. A candidate is released only if the
   worst-sample fidelity gain, shape and pump-overlap guards all pass and the
   frames are consistent with the model. Otherwise the controller holds.

The certificate is conditional on the frozen passive model family (SLM-plane
aberration, SLM gain, disk phase/log-gain screen) and its empirical
calibration. It is not a full amplifier/thermal-solver guarantee. Fresh solver
replay (``examples/replay_ybyag_nn_candidate.py``) and measured verification
after application remain required. Importing this module launches nothing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as tnf

from hoyag.propagation import Grid2D, angular_spectrum_transfer
from ybluag.beam_shaping import gaussian_seed_and_target_mask
from ybyag_dataset.modal_teacher import modal_basis
from ybluag.multipass_geometry import from_nominal as multipass_from_nominal
from ybyag_control.nn_v2 import GateLimits, verify_measured_outcome  # noqa: F401 (re-export)

SCHEMA = 'ybyag_controller_v3_oneshot_1'
TARGETS = ('Gaussian TEM00', 'Flattop super-Gaussian', 'Helical LG(0,+1)',
           'Needle Bessel-Gaussian')
WAVELENGTH_M = 1030e-9
SLM_MODES = 14
DISK_MODES = 14
RESIDUAL = 12
STATE = SLM_MODES + DISK_MODES + 2
# Network state outputs are divided by these physical scales.
STATE_SCALE = np.r_[np.ones(SLM_MODES), np.ones(DISK_MODES), .02, .1]
# Weak MAP prior for refinement; wider than the sampled training ranges.
PRIOR_SIGMA = np.r_[np.full(SLM_MODES, 2.), np.full(DISK_MODES, 2.), .04, .4]
CONTEXT = 10
CHANNELS = 11
# Intensity at two planes cannot tell phi(x) from its twin -phi(-x) when the
# diversity is weak. Zernike Z_n^m maps to -(-1)^n Z_n^m, so even-n modes flip.
_PARITY = np.array([1. if n % 2 else -1. for n in range(1, 5) for _ in range(n+1)])
TWIN = np.r_[_PARITY, _PARITY, 1., 1.]
TWINS = (TWIN, np.r_[_PARITY, np.ones(DISK_MODES), 1., 1.],
         np.r_[np.ones(SLM_MODES), _PARITY, 1., 1.])


@dataclass(frozen=True)
class Spec:
    """Nominal optical design; everything here is known, never hidden truth."""
    slm_n: int = 384
    field_m: float = 12e-3
    factor: int = 2
    waist_m: float = .6e-3
    pump_radius_m: float = 1e-3
    slm_to_disk_m: float = .25
    output_distance_m: float = 0.
    planes_m: tuple = (0., 1.)
    crop: int = 96
    slm_bits: int = 8
    crosstalk_sigma_px: float = .35
    camera_width_px: int = 1920
    camera_height_px: int = 1080
    camera_fov_width_m: float = 10e-3
    # Multipass geometry. With a nonzero inter-pass path the disk screen is
    # split over ``encounters`` crossings with diffraction between them.
    encounters: int = 10
    inter_pass_distance_m: float = 0.
    inter_pass_focal_m: float = 0.
    multipass: dict = None            # mirror-array layout (nominal['multipass'])

    def __post_init__(self):
        if self.slm_n % self.factor or self.factor < 1:
            raise ValueError('SLM grid must be an integer multiple of the optical grid')
        if self.crop % 16 or not 16 <= self.crop <= self.slm_n // self.factor:
            raise ValueError('network crop must be a multiple of 16 inside the optical grid')
        if len(self.planes_m) != 2:
            raise ValueError('two phase-diverse camera planes are required')

    @property
    def opt_n(self):
        return self.slm_n // self.factor

    def slm_grid(self):
        return Grid2D.square(self.slm_n, self.field_m)

    def opt_grid(self):
        return Grid2D.square(self.opt_n, self.field_m)

    @classmethod
    def from_config(cls, config, **overrides):
        nominal, ranges, camera = config['nominal'], config['ranges'], config['camera']
        values = dict(slm_n=int(nominal['grid_n']),
            field_m=float(nominal['field_size_mm'])*1e-3,
            waist_m=float(nominal['waist_mm'])*1e-3,
            pump_radius_m=float(nominal['radius_mm'])*1e-3,
            slm_to_disk_m=float(nominal['slm_to_disk_m']),
            output_distance_m=float(nominal['output_distance_m']),
            planes_m=tuple(float(z) for z in config['planes_m']),
            slm_bits=int(ranges['slm_bits']),
            crosstalk_sigma_px=float(ranges['slm_crosstalk_sigma_pixels']),
            camera_width_px=int(camera['width']), camera_height_px=int(camera['height']),
            camera_fov_width_m=float(camera['object_fov_width_mm'])*1e-3,
            encounters=(2*int(nominal['regen_round_trips']) if nominal['architecture'] == 'regenerative'
                        else int(nominal['signal_traversals'])),
            inter_pass_distance_m=float(nominal.get('inter_pass_distance_m', 0.)),
            inter_pass_focal_m=float(nominal.get('inter_pass_focal_m', 0.)),
            multipass=nominal.get('multipass'))
        values.update(overrides)
        return cls(**values)

    def geometry(self):
        """Nominal multipass geometry (no mirror errors), or None for the ideal relay."""
        nominal = dict(multipass=self.multipass, inter_pass_distance_m=self.inter_pass_distance_m,
                       inter_pass_focal_m=self.inter_pass_focal_m)
        return multipass_from_nominal(nominal, self.encounters, waist_m=self.waist_m,
                                      wavelength_m=WAVELENGTH_M)

    def to_json(self):
        return {**asdict(self), 'planes_m': list(self.planes_m)}

    @classmethod
    def from_json(cls, data):
        return cls(**{**data, 'planes_m': tuple(data['planes_m'])})


def block_mean(a, factor):
    """Area average by an integer factor; keeps Grid2D cell-centre alignment."""
    if factor == 1:
        return a
    *lead, n, m = a.shape
    return a.reshape(*lead, n//factor, factor, m//factor, factor).mean(axis=(-3, -1))


def _gaussian_kernel(sigma):
    radius = int(4*sigma+.5)
    x = np.arange(-radius, radius+1, dtype=float)
    k = np.exp(-.5*(x/sigma)**2)
    return k/k.sum()


class Optics:
    """Differentiable frozen passive model shared by training and inference.

    SLM-plane maps live on the native SLM grid. The field is area-averaged to
    the optical grid, propagated to the disk, multiplied by the disk
    phase/log-gain screen, then propagated to the output and camera planes.
    """

    def __init__(self, spec: Spec, device='cpu', precision='single'):
        self.spec = spec
        self.device = torch.device(device)
        self.real = torch.float32 if precision == 'single' else torch.float64
        self.complex = torch.complex64 if precision == 'single' else torch.complex128
        grid, slm = spec.opt_grid(), spec.slm_grid()
        self.dx = grid.dx
        t = self.tensor
        self.geometry = spec.geometry()
        paths = [] if self.geometry is None else self.geometry.paths
        halves = [p.length_m/2 for p in paths]
        free = [h for h, p in zip(halves, paths) if p.kind == 'free_space']
        focus = [p.defocus_m for p in paths if p.kind == 'image']
        distances = ({spec.slm_to_disk_m, spec.output_distance_m, *free, *focus, *spec.planes_m}
                     - {0.})
        self.transfer = {d: torch.as_tensor(angular_spectrum_transfer(grid, WAVELENGTH_M, d),
                                            dtype=self.complex, device=self.device)
                         for d in distances}
        xs, ys = slm.mesh
        x, y = grid.mesh
        self.xs, self.ys, self.x, self.y = t(xs), t(ys), t(x), t(y)
        w = spec.waist_m
        _, basis, weight = modal_basis(xs, ys, np.exp(-(xs*xs+ys*ys)/w**2), w, radial_order=4)
        if len(basis) != SLM_MODES:
            raise ValueError('unexpected SLM modal basis')
        self.slm_basis_np = basis
        self.slm_basis, self.slm_weight = t(basis), t(weight)
        self.slm_basis_opt = t(block_mean(basis, spec.factor))
        rayleigh = np.pi*w*w/WAVELENGTH_M
        wide = 1.25*w*np.sqrt(1+(spec.slm_to_disk_m/rayleigh)**2)
        _, disk, disk_weight = modal_basis(x, y, np.exp(-(x*x+y*y)/wide**2), wide, radial_order=4)
        if len(disk) != DISK_MODES:
            raise ValueError('unexpected disk modal basis')
        self.disk_basis, self.disk_weight = t(disk), t(disk_weight)
        half = 2.4*w
        m = 2*int(round(half/slm.dx))
        start = (spec.slm_n-m)//2
        self.residual_window = (start, start+m)
        self.residual_taper = t(np.exp(-(np.hypot(xs, ys)/(2.2*w))**8))
        if spec.crosstalk_sigma_px > 0:
            k = _gaussian_kernel(spec.crosstalk_sigma_px)
            self.crosstalk = t(k)
        else:
            self.crosstalk = None
        c0 = (spec.opt_n-spec.crop)//2
        self.crop = slice(c0, c0+spec.crop)
        g = self.geometry
        self.passes = 1 if g is None else g.encounters
        if g is not None:
            # Bilinear sampling of the disk map at each encounter's incidence
            # (identical to map_coordinates(order=1, mode='nearest')).
            n = spec.opt_n
            samples = []
            for k in range(g.encounters):
                rows, cols = g.sampling(k, grid)
                samples.append(np.stack([2*cols/(n-1)-1, 2*rows/(n-1)-1], -1))
            self.encounter_grid = t(np.stack(samples))
            self.encounter_scale = [g.internal_scale(k) for k in range(g.encounters)]
            self.mirrors = torch.as_tensor(np.stack([g.mirror_phase(k, grid, WAVELENGTH_M,
                include_tilt=False) for k in range(len(g.paths))]) if g.paths else
                np.zeros((0, *grid.shape)), dtype=self.complex, device=self.device)
            self.halves = halves
            fx, fy = np.meshgrid(grid.fx, grid.fy, indexing='xy')
            self.fx, self.fy = t(fx), t(fy)
            # Longitudinal wavenumber for per-sample relay focus errors.
            k = 2*math.pi/WAVELENGTH_M
            q = (2*math.pi)**2*(fx*fx+fy*fy)
            self.kz = t(np.sqrt(np.maximum(k*k-q, 0.)))
            self.propagating = t(q <= k*k)

    def tensor(self, a, dtype=None):
        return torch.as_tensor(np.asarray(a), dtype=dtype or self.real, device=self.device)

    def block(self, u):
        f = self.spec.factor
        if f == 1:
            return u
        *lead, n, m = u.shape
        return u.reshape(*lead, n//f, f, m//f, f).mean(dim=(-3, -1))

    def propagate(self, u, distance):
        if distance == 0:
            return u
        return torch.fft.ifft2(torch.fft.fft2(u)*self.transfer[distance])

    def step_map(self, coefficients, residual=None):
        """Phase step on the SLM grid from modal coefficients and 12x12 residual."""
        step = torch.einsum('bk,kij->bij', coefficients, self.slm_basis)
        if residual is not None:
            a, b = self.residual_window
            up = tnf.interpolate(residual[:, None], size=(b-a, b-a), mode='bicubic',
                                 align_corners=True)[:, 0]
            full = torch.zeros_like(step)
            full[:, a:b, a:b] = up
            full = full*self.residual_taper
            full = full-(full*self.slm_weight).sum((-2, -1), keepdim=True)
            step = step+full
        return step

    def _blur(self, a):
        k = self.crosstalk
        r = (len(k)-1)//2
        a = a[:, None]
        a = tnf.conv2d(tnf.pad(a, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
        a = tnf.conv2d(tnf.pad(a, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1))
        return a[:, 0]

    def render(self, command, gain, spatial=None, *, quantize=True, crosstalk=True):
        """Unit-modulus SLM phasor; matches ``apply_slm`` for uniform gain maps.

        Wrapping and quantization use straight-through gradients.
        """
        two_pi = 2*math.pi
        wrapped = command-two_pi*torch.floor(command/two_pi).detach()
        bits = self.spec.slm_bits
        if quantize and bits:
            levels = 2**bits
            q = torch.round(wrapped/two_pi*(levels-1))*two_pi/(levels-1)
            wrapped = wrapped+(q-wrapped).detach()
        scale = (1+gain).view(-1, 1, 1)
        if spatial is not None:
            scale = scale*spatial
        phase = scale*wrapped
        if crosstalk and self.crosstalk is not None:
            re, im = self._blur(torch.cos(phase)), self._blur(torch.sin(phase))
            norm = torch.sqrt(re*re+im*im+1e-30)
            return torch.complex(re/norm, im/norm)
        return torch.polar(torch.ones_like(phase), phase)

    def encounter_screen(self, k, disk_map):
        """Per-encounter share of a total disk map as seen at encounter k."""
        share = disk_map/self.passes
        if self.geometry is None:
            return share
        grid = self.encounter_grid[k].expand(len(share), -1, -1, -1)
        sampled = tnf.grid_sample(share[:, None], grid, mode='bilinear', padding_mode='border',
                                  align_corners=True)[:, 0]
        return sampled*self.encounter_scale[k]

    def _image_relay(self, k, d, tilts=None, defocus=None):
        """4f image relay: point inversion, fold-mirror image shift, focus error.

        ``defocus`` (B, gaps) adds a per-sample path-length error to the relay's
        nominal one (exact angular-spectrum transfer).
        """
        path = self.geometry.paths[k]
        d = torch.flip(d, dims=(-2, -1))
        if tilts is not None or defocus is not None:
            phase = torch.zeros_like(d.real)
            if tilts is not None:
                dx = (2*path.focal_m*tilts[:, k, 0]).view(-1, 1, 1)
                dy = (2*path.focal_m*tilts[:, k, 1]).view(-1, 1, 1)
                phase = phase-2*math.pi*(self.fx*dx+self.fy*dy)
            if defocus is not None:
                phase = phase+self.kz*defocus[:, k].view(-1, 1, 1)
            transfer = torch.polar(self.propagating if defocus is not None else torch.ones_like(phase), phase)
            d = torch.fft.ifft2(torch.fft.fft2(d)*transfer)
        return self.propagate(d, path.defocus_m)

    def relay(self, k, d, tilts=None, defocus=None):
        if self.geometry.paths[k].kind == 'reflection':
            return d                      # HR back reflection inside one bounce
        if self.geometry.paths[k].kind == 'image':
            return self._image_relay(k, d, tilts, defocus)
        """Encounter k -> k+1: half path, array mirror (+ pointing error), half path."""
        d = self.propagate(d, self.halves[k])*self.mirrors[k]
        if tilts is not None:
            wave = 2*math.pi/WAVELENGTH_M
            tilt = 2*wave*(tilts[:, k, 0].view(-1, 1, 1)*self.x+tilts[:, k, 1].view(-1, 1, 1)*self.y)
            d = d*torch.polar(torch.ones_like(tilt), tilt)
        return self.propagate(d, self.halves[k])

    def forward(self, amp, phasor, slm_phase=None, disk_phase=None, disk_logamp=None,
                *, planes=True, tilts=None, defocus=None):
        u = self.block(amp*phasor)
        if slm_phase is not None:
            u = u*torch.polar(torch.ones_like(slm_phase), slm_phase)
        d = self.propagate(u, self.spec.slm_to_disk_m)
        # Totals over all encounters, split equally per crossing. The phase is
        # seen at each encounter's incidence; gain stays in beam coordinates,
        # as in the solver.
        gain = None if disk_logamp is None else torch.exp(disk_logamp/self.passes)
        for k in range(self.passes):
            if k:
                d = self.relay(k-1, d, tilts, defocus)
            if disk_phase is not None:
                ph = self.encounter_screen(k, disk_phase)
                d = d*torch.polar(torch.ones_like(ph) if gain is None else gain, ph)
            elif gain is not None:
                d = d*gain
        o = self.propagate(d, self.spec.output_distance_m)
        out = dict(slm=u, disk=d, output=o)
        if planes:
            out['planes'] = torch.stack([self.propagate(o, z) for z in self.spec.planes_m], 1)
        return out

    def pump_profile(self, radius_m, center_m=None):
        radius = torch.as_tensor(radius_m, dtype=self.real, device=self.device).view(-1, 1, 1)
        cx = cy = 0.
        if center_m is not None:
            c = torch.as_tensor(center_m, dtype=self.real, device=self.device).view(-1, 2)
            cx, cy = c[:, 0].view(-1, 1, 1), c[:, 1].view(-1, 1, 1)
        return torch.exp(-2*((self.x-cx)**2+(self.y-cy)**2)/radius**2)

    def screens(self, theta, pump):
        """Physical state (B,30) -> SLM gain, SLM-plane phase, disk phase, log-gain."""
        a_s, a_d = theta[:, :SLM_MODES], theta[:, SLM_MODES:SLM_MODES+DISK_MODES]
        gain, gamma = theta[:, -2], theta[:, -1]
        slm_phase = torch.einsum('bk,kij->bij', a_s, self.slm_basis_opt)
        disk_phase = torch.einsum('bk,kij->bij', a_d, self.disk_basis)
        centred = pump-(pump*self.disk_weight).sum((-2, -1), keepdim=True)
        return gain, slm_phase, disk_phase, gamma.view(-1, 1, 1)*centred

    def model(self, theta, amp, command, pump, *, planes=True, step=None, quantize=True):
        gain, slm_phase, disk_phase, logamp = self.screens(theta, pump)
        cmd = command if step is None else command+step
        phasor = self.render(cmd, gain, quantize=quantize)
        return self.forward(amp, phasor, slm_phase, disk_phase, logamp, planes=planes)

    def desired(self, waist_m, target_phase):
        """Nominal centred Gaussian with the intended mask, as the dataset defines it."""
        w = torch.as_tensor(waist_m, dtype=self.real, device=self.device).view(-1, 1, 1)
        source = torch.exp(-(self.xs**2+self.ys**2)/w**2)
        phasor = torch.polar(torch.ones_like(target_phase), target_phase)
        return self.forward(source, phasor)


def overlaps(output, desired, disk, pump):
    """Coherent fidelity, amplitude-shape overlap and pump overlap (batched)."""
    dims = (-2, -1)
    po, pd = output.real**2+output.imag**2, desired.real**2+desired.imag**2
    no, nd = po.sum(dims), pd.sum(dims)
    inner = (desired.conj()*output).sum(dims)
    fidelity = (inner.real**2+inner.imag**2)/(no*nd)
    shape = (torch.sqrt(po+1e-30)*torch.sqrt(pd+1e-30)).sum(dims)**2/(no*nd)
    pdisk = disk.real**2+disk.imag**2
    pump_overlap = (pdisk*pump).sum(dims)/pdisk.sum(dims)
    return fidelity, shape, pump_overlap


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------

@dataclass
class TrialInputs:
    """Measured/nominal inputs for one trial on the V3 grids (numpy)."""
    family: str
    waist_m: float
    pump_radius_m: float
    amp: np.ndarray          # SLM grid, incoming amplitude (max 1)
    command: np.ndarray      # SLM grid, delivered command (rad)
    target: np.ndarray       # SLM grid, intended structured mask (rad)
    meas: np.ndarray         # (2,n,n) mean photoelectrons per optical cell
    var: np.ndarray          # (2,n,n) variance of that mean
    valid: np.ndarray        # (2,n,n) bool
    context: np.ndarray      # (CONTEXT,)

    @classmethod
    def from_trial(cls, measured, metadata, spec: Spec, affine=None):
        """Only ``input__*`` measurements, camera settings and nominal design."""
        incoming = np.asarray(measured['input__incoming_beam_shape'], float)
        command = np.asarray(measured['input__slm_command_rad'], float)
        target = np.asarray(measured['input__target_phase_mask_rad'], float)
        for a in (incoming, command, target):
            if a.shape != (spec.slm_n, spec.slm_n) or not np.all(np.isfinite(a)):
                raise ValueError('measured SLM-plane maps must match the configured grid')
        if incoming.max() <= 0 or np.any(incoming < 0):
            raise ValueError('invalid measured input beam')
        cs = metadata['camera_settings']
        meas, var, valid, saturated = bin_camera(measured['input__camera_adu'], cs, spec,
            affine=affine if affine is not None else cs.get('object_to_pixel_affine'))
        waist = float(measured['input__seed_waist_m'])
        pump = float(measured['input__pump_radius_m'])
        family = metadata['selected_target']
        cell = (spec.opt_grid().dx*spec.camera_width_px/spec.camera_fov_width_m)**2
        context = context_vector(family, waist, pump, spec,
                                 (meas*valid).sum((1, 2))*cell, saturated)
        return cls(family, waist, pump, np.sqrt(incoming/incoming.max()), command, target,
                   meas, var, valid, context)


def context_vector(family, waist_m, pump_radius_m, spec, total_electrons, saturated):
    total = np.maximum(np.asarray(total_electrons, float), 1.)
    return np.r_[np.eye(4)[TARGETS.index(family)],
                 10*(waist_m/spec.waist_m-1), 10*(pump_radius_m/spec.pump_radius_m-1),
                 np.log10(total)-8, 10*np.asarray(saturated, float)].astype(np.float32)


def bin_camera(adu, cs, spec: Spec, affine=None):
    """Area-bin calibrated camera frames onto optical cells.

    Returns mean photoelectrons, variance of that mean (shot, read, dark,
    quantization and PRNU), cell validity, and per-plane saturated fraction.
    Saturated or hot pixels invalidate/are excluded; nothing is recentred.
    """
    adu = np.asarray(adu, float)
    if adu.ndim != 3 or adu.shape[0] != 2 or not np.all(np.isfinite(adu)):
        raise ValueError('two finite phase-diverse camera frames required')
    _, h, w = adu.shape
    max_adu = 2**int(cs['adc_bits'])-1
    black, full = float(cs['black_level_adu']), float(cs['full_well_e'])
    gain = full/(max_adu-black)
    electrons = np.maximum(adu-black, 0)*gain
    saturated = adu >= max_adu
    pixel = float(cs['object_fov_width_mm'])*1e-3/w
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    if affine is None:
        ox, oy = (cols+.5-w/2)*pixel, (rows+.5-h/2)*pixel
    else:
        affine = np.asarray(affine, float)
        if affine.shape != (2, 3) or not np.all(np.isfinite(affine)):
            raise ValueError('invalid camera registration')
        inverse = np.linalg.inv(affine[:, :2])
        ox, oy = np.einsum('ij,jhw->ihw', inverse,
                           np.stack([cols-affine[0, 2], rows-affine[1, 2]]))
    grid = spec.opt_grid()
    n = grid.nx
    ix = np.floor((ox-(grid.x[0]-grid.dx/2))/grid.dx).astype(int)
    iy = np.floor((oy-(grid.y[0]-grid.dy/2))/grid.dy).astype(int)
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    cell = np.where(inside, iy*n+ix, 0).ravel()
    per_cell = np.bincount(cell, inside.ravel().astype(float), n*n)
    read = float(cs['read_noise_e'])
    dark = float(cs['dark_current_e_s'])*float(cs['exposure_s'])+float(cs.get('background_e', 0))
    extra = read**2+float(cs.get('dsnu_rms_e', 0))**2+gain**2/12
    prnu = float(cs.get('prnu_rms', 0))
    meas, var, valid, fraction = [], [], [], []
    from scipy.ndimage import median_filter
    for e, sat in zip(electrons, saturated):
        local = median_filter(e, 3, mode='nearest')
        hot = e > local+8*np.sqrt(local+dark+extra)+.05*full
        use = (inside & ~hot).ravel().astype(float)
        count = np.bincount(cell, use, n*n)
        total = np.bincount(cell, e.ravel()*use, n*n)
        bad = np.bincount(cell, (sat & inside).ravel().astype(float), n*n)
        mean = total/np.maximum(count, 1)
        pixel_var = np.maximum(mean, 0)+dark+extra+(prnu*mean)**2
        ok = (count >= .75*np.maximum(per_cell, 1)) & (per_cell > 0) & (bad == 0)
        meas.append(mean.reshape(n, n))
        var.append((pixel_var/np.maximum(count, 1)).reshape(n, n))
        valid.append(ok.reshape(n, n))
        fraction.append(float(sat[inside].mean()))
    return (np.stack(meas).astype(np.float32), np.stack(var).astype(np.float32),
            np.stack(valid), np.asarray(fraction))


def amplitude_fit(intensity, meas, var, valid, model_floor=.01):
    """Scale-free amplitude residual between modelled planes and measurement.

    ``intensity``: (B,2,n,n) modelled |E|^2; the unknown common camera scale is
    eliminated in closed form. Returns whitened residuals, chi2, relative RMS.
    """
    m = torch.sqrt(torch.clamp(meas, min=0.))
    a = torch.sqrt(intensity+1e-30)
    peak = (m*valid).amax((-3, -2, -1), keepdim=True)
    sigma2 = var/(4*torch.clamp(meas, min=1.))+(model_floor*peak)**2
    w = valid/sigma2
    scale = (w*m*a).sum((-3, -2, -1), keepdim=True)/(w*a*a).sum((-3, -2, -1), keepdim=True)
    diff = m-scale*a
    residual = torch.sqrt(w)*diff
    chi2 = (residual**2).sum((-3, -2, -1))
    relative = torch.sqrt((valid*diff**2).sum((-3, -2, -1))/(valid*m*m).sum((-3, -2, -1)))
    return residual, chi2, relative


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------

def network_inputs(optics: Optics, batch):
    """Image channels and context. ``batch`` holds tensors on optics.device."""
    with torch.no_grad():
        amp, command = batch['amp'], batch['command']
        zero = torch.zeros(len(amp), dtype=optics.real, device=optics.device)
        phasor = optics.render(command, zero)
        nominal = optics.forward(amp, phasor)
        valid = batch['valid'].to(optics.real)
        m = torch.sqrt(torch.clamp(batch['meas'], min=0.))*valid
        m = m/torch.clamp(m.amax((-3, -2, -1), keepdim=True), min=1e-12)

        def norm(a):
            return a/torch.clamp(a.amax((-3, -2, -1), keepdim=True), min=1e-12)
        nom = norm(nominal['planes'].abs())
        des = norm(batch['desired_planes'].abs())
        u = nominal['slm']
        u = u/torch.clamp(u.abs().amax((-2, -1), keepdim=True), min=1e-12)
        a = norm(optics.block(amp)[:, None])[:, 0]
        images = torch.cat([m, valid, nom, des, u.real[:, None], u.imag[:, None], a[:, None]], 1)
        c = optics.crop
        return images[..., c, c].contiguous(), batch['context']


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout, context):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, 2, 1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1, 2)
        self.film = nn.Linear(context, 2*cout)

    def forward(self, x, c):
        h = self.norm1(self.conv1(x))
        gamma, beta = self.film(c).chunk(2, -1)
        h = tnf.silu(h*(1+gamma[..., None, None])+beta[..., None, None])
        return tnf.silu(self.norm2(self.conv2(h))+self.skip(x))


class OneShotNet(nn.Module):
    """Measured frames -> full correction step and hidden-state estimate."""

    def __init__(self, crop=96, widths=(32, 64, 96, 128), hidden=384):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(CONTEXT, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.stem = nn.Sequential(nn.Conv2d(CHANNELS, widths[0], 3, 1, 1),
                                  nn.GroupNorm(8, widths[0]), nn.SiLU())
        chans = (widths[0], *widths)
        self.blocks = nn.ModuleList(FiLMBlock(a, b, 64) for a, b in zip(chans[:-1], chans[1:]))
        side = crop//2**len(widths)
        self.trunk = nn.Sequential(nn.Linear(widths[-1]*side*side+64, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.state_mean = nn.Linear(hidden, STATE)
        self.state_logvar = nn.Linear(hidden, STATE)
        self.correction = nn.Linear(hidden, SLM_MODES+RESIDUAL**2)
        # Starts exactly at hold; the outcome loss moves it away.
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.state_logvar.weight)

    def forward(self, images, context):
        c = self.context(context)
        x = self.stem(images)
        for block in self.blocks:
            x = block(x, c)
        h = self.trunk(torch.cat([x.flatten(1), c], 1))
        corr = self.correction(h)
        return dict(state_mean=self.state_mean(h),
                    state_logvar=torch.clamp(self.state_logvar(h), -12, 6),
                    coefficients=corr[:, :SLM_MODES],
                    residual=.5*corr[:, SLM_MODES:].view(-1, RESIDUAL, RESIDUAL))


# ----------------------------------------------------------------------------
# Synthetic passive plant for label-free training
# ----------------------------------------------------------------------------

def _correlated(shape, sigma_px, generator, device, dtype):
    """Batched Gaussian-correlated unit-RMS maps (periodic FFT filter)."""
    b, n, m = shape
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    fy = torch.fft.fftfreq(n, device=device, dtype=dtype)[:, None]
    fx = torch.fft.fftfreq(m, device=device, dtype=dtype)[None]
    kernel = torch.exp(-2*(math.pi*sigma_px)**2*(fx*fx+fy*fy))
    field = torch.fft.ifft2(torch.fft.fft2(noise)*kernel).real
    field = field-field.mean((-2, -1), keepdim=True)
    return field/torch.clamp(field.square().mean((-2, -1), keepdim=True).sqrt(), min=1e-12)


class SyntheticPlant:
    """Samples passive optical states from the dataset's distortion ranges.

    Mirrors the generator's seed/aperture/tilt, upstream Zernike + residual
    screen, SLM gain/quantization/crosstalk, disk thickness/surface screen,
    a pump-shaped thermal phase and log-gain, and cell-binned camera noise.
    ``widen`` > 1 deliberately broadens every range for robustness. It is a
    frozen passive proxy, not the pulsed amplifier/thermal solver.
    """

    def __init__(self, spec: Spec, config, device='cpu', *, widen=1.25,
                 thermal_peak_rad=1.0, log_gain_max=.3, waist_lattice=9):
        self.spec, self.config, self.widen = spec, config, float(widen)
        self.optics = Optics(spec, device, 'single')
        o = self.optics
        self.device = o.device
        ranges, nominal = config['ranges'], config['nominal']
        self.ranges, self.nominal = ranges, nominal
        self.thermal_peak = float(thermal_peak_rad)
        self.log_gain_max = float(log_gain_max)
        slm = spec.slm_grid()
        span = ranges['seed_waist_fraction']
        self.waists = spec.waist_m*(1+np.linspace(-span, span, waist_lattice))
        masks, desired = [], []
        for family in TARGETS:
            row, drow = [], []
            for w in self.waists:
                _, phase = gaussian_seed_and_target_mask(slm, float(w), 1., family,
                    WAVELENGTH_M, spec.slm_to_disk_m)
                t = o.tensor(phase)[None]
                row.append(t[0])
                drow.append(o.desired(float(w), t)['planes'][0])
            masks.append(torch.stack(row))
            desired.append(torch.stack(drow))
        self.masks = torch.stack(masks)          # (4,W,N,N)
        self.desired_planes = torch.stack(desired)  # (4,W,2,n,n)
        xs, ys = slm.mesh
        r = np.hypot(xs, ys)/5e-3
        th = np.arctan2(ys, xs)
        zern = [r*np.cos(th), r*np.sin(th), 2*r*r-1, r*r*np.cos(2*th), r*r*np.sin(2*th),
                (3*r**3-2*r)*np.cos(th), (3*r**3-2*r)*np.sin(th), 6*r**4-6*r*r+1,
                r**3*np.cos(3*th), r**3*np.sin(3*th), (4*r**4-3*r*r)*np.cos(2*th)]
        self.external_basis = o.tensor(np.stack(zern))
        self.external_inside = o.tensor(r <= 1)
        radius = float(np.max(np.abs(slm.x)))
        self.search_modes = o.tensor(np.stack([(xs/radius)**2+(ys/radius)**2,
            (xs*xs-ys*ys)/radius**2, 2*xs*ys/radius**2, xs/radius, ys/radius]))
        from ybyag.model import YbYAGMaterial
        self.index_minus_one = YbYAGMaterial(yb_at_percent=10.).cavity_phase_index-1
        self.traversals = (2*int(nominal['regen_round_trips'])
                           if nominal['architecture'] == 'regenerative'
                           else int(nominal['signal_traversals']))
        self.thickness_m = float(nominal['disk_thickness_um'])*1e-6
        cam = config['camera']
        self.camera = cam
        self.pixel = spec.camera_fov_width_m/spec.camera_width_px
        self.pixels_per_cell = (o.dx/self.pixel)**2
        half_w = spec.camera_fov_width_m/2
        half_h = self.pixel*spec.camera_height_px/2
        self.fov = ((o.x.abs() <= half_w) & (o.y.abs() <= half_h)).to(o.real)
        self.disk_mask = o.tensor(np.hypot(xs, ys) <= float(nominal['disk_radius_mm'])*1e-3)
        # Measured relative Yb maps (sign-corrected), a bank of seeded placements
        # per configured concentration. They modulate the disk log-gain.
        self.yb_maps = None
        if ranges.get('yb_distribution', 'random') == 'measured':
            from ybyag_dataset.distortions.measured_doping import relative_map, sampled_placement
            rng = np.random.default_rng(20260930)
            grid = spec.opt_grid()
            maps = []
            for at in sorted(set(float(c) for c in nominal['yb_at_percent_candidates'])):
                for _ in range(48):
                    offset, rotation = sampled_placement(ranges, rng, stress=self.widen)
                    maps.append(relative_map(grid, at, offset_m=offset, rotation_rad=rotation,
                        deviation_scale=float(ranges.get('yb_map_deviation_scale', -1.7)),
                        smooth_mm=float(ranges.get('yb_map_smooth_mm', .1)),
                        taper_mm=float(ranges.get('yb_map_taper_mm', .5)),
                        edge_trim_mm=float(ranges.get('yb_map_edge_trim_mm', .3)),
                        nearest=bool(ranges.get('yb_map_nearest_sample', False))))
            self.yb_maps = o.tensor(np.stack(maps))
        # Net log-amplitude per unit relative Yb (lossy at 0.2 W, gain at higher pump).
        self.yb_log_gain_max = .4

    def _u(self, a, b, shape, g):
        return a+(b-a)*torch.rand(shape, generator=g, device=self.device, dtype=self.optics.real)

    def _n(self, shape, g):
        return torch.randn(shape, generator=g, device=self.device, dtype=self.optics.real)

    def sample(self, batch, generator):
        o, g, R, s = self.optics, generator, self.ranges, self.widen
        B, N = batch, self.spec.slm_n
        dev, real = self.device, o.real
        fam = torch.randint(0, 4, (B,), generator=g, device=dev)
        wi = torch.randint(0, len(self.waists), (B,), generator=g, device=dev)
        waist = o.tensor(self.waists)[wi]
        target = self.masks[fam, wi]
        desired = self.desired_planes[fam, wi]
        v = lambda t: t.view(-1, 1, 1)
        ell = 1+s*self._u(-1, 1, (B,), g)*R['seed_ellipticity_fraction']
        wx, wy = waist*ell.sqrt(), waist/ell.sqrt()
        shift = s*self._u(0, 1, (B,), g)*R['seed_offset_radius_fraction']*self.spec.pump_radius_m
        ang = self._u(0, 2*math.pi, (B,), g)
        cx, cy = shift*torch.cos(ang), shift*torch.sin(ang)
        amp = torch.exp(-((o.xs-v(cx))**2/v(wx)**2+(o.ys-v(cy))**2/v(wy)**2))
        lo, hi = R['seed_aperture_radius_waists']
        ar = self._u(lo, hi, (B,), g)*waist
        ad = self._n((B, 2), g)*s*R['seed_aperture_decenter_waist_fraction']*self.spec.waist_m
        amp = amp*(((o.xs-v(ad[:, 0]))**2+(o.ys-v(ad[:, 1]))**2) <= v(ar)**2)
        k = 2*math.pi/WAVELENGTH_M
        tilt = self._n((B, 2), g)*s*R['seed_angle_urad']*1e-6
        upstream = k*(v(tilt[:, 0])*o.xs+v(tilt[:, 1])*o.ys)
        coeff = self._n((B, len(self.external_basis)), g)
        z = torch.einsum('bk,kij->bij', coeff, self.external_basis)
        inside = self.external_inside
        z = z-v((z*inside).sum((-2, -1))/inside.sum())
        rms = v(((z*inside)**2).sum((-2, -1))/inside.sum()).sqrt()
        level = s*self._u(0, 1, (B,), g)*R['external_aberration_rms_waves']
        residual = _correlated((B, N, N), max(2, N/12), g, dev, real)
        upstream = upstream+2*math.pi*(z*v(level)/rms+s*R['external_residual_rms_waves']*residual)
        gain = self._n((B,), g)*s*R['slm_global_gain_fraction']
        spatial = (1+s*R['slm_spatial_gain_rms_fraction']*_correlated((B, N, N), max(2, N/8), g, dev, real))
        spatial = spatial*(1+s*R['slm_pixel_gain_rms_fraction']*self._n((B, N, N), g))
        # Current command: first trial, coordinate-search trial, or a previous
        # partial/overshooting correction (teaches hold and second iterations).
        truth_slm = torch.einsum('kij,bij->bk', o.slm_basis, o.slm_weight*upstream)
        smooth = torch.einsum('bk,kij->bij', truth_slm, o.slm_basis)
        kind = self._u(0, 1, (B,), g)
        search = self.search_modes[torch.randint(0, 5, (B,), generator=g, device=dev)]
        sign = torch.where(self._u(0, 1, (B,), g) < .5, -1., 1.).to(real)
        previous = -v(self._u(.3, 1.15, (B,), g))*smooth
        jitter = torch.einsum('bk,kij->bij', self._n((B, SLM_MODES), g)*v(self._u(0, .12, (B,), g))[:, :, 0],
                              o.slm_basis)
        delta = torch.where(v(kind) < .35, torch.zeros_like(target),
                torch.where(v(kind) < .6, v(sign)*.25*search, previous+jitter))
        command = torch.remainder(target+delta, 2*math.pi)
        # Disk screen on the optical grid.
        n = self.spec.opt_n
        f = self.spec.factor
        thick = self._u(.5, 1.5, (B,), g)*s*R['thickness_rms_fraction']*self.thickness_m
        surface = s*R['surface_figure_rms_nm']*1e-9
        per = self.traversals*k
        disk_phase = (per*self.index_minus_one*v(thick)*_correlated((B, n, n), max(2, N/7)/f, g, dev, real) +
                      per*surface*_correlated((B, n, n), max(2, N/7)/f, g, dev, real))
        prad = self.spec.pump_radius_m*(1+s*self._u(-1, 1, (B,), g)*R['pump_radius_fraction'])
        pshift = s*self._u(0, 1, (B,), g)*R['pump_offset_radius_fraction']*prad
        pang = self._u(0, 2*math.pi, (B,), g)
        pcenter = torch.stack([pshift*torch.cos(pang), pshift*torch.sin(pang)], 1)
        pump = o.pump_profile(prad, pcenter)
        disk_phase = disk_phase+v(self._u(-1, 1, (B,), g)*self.thermal_peak)*pump
        gamma = self._u(0, 1, (B,), g)*self.log_gain_max
        tilt_sigma = s*float(R.get('mirror_tilt_error_urad', 0.))*1e-6
        tilts = (self._n((B, max(o.passes-1, 0), 2), g)*tilt_sigma
                 if o.geometry is not None and self.spec.multipass else None)
        logamp = v(gamma)*(pump-(pump*o.disk_weight).sum((-2, -1), keepdim=True))
        if self.yb_maps is not None:
            pick = torch.randint(0, len(self.yb_maps), (B,), generator=g, device=dev)
            yb_gain = self._u(-1, 1, (B,), g)*self.yb_log_gain_max
            logamp = logamp+v(yb_gain)*(self.yb_maps[pick]-1)
        focus_sigma = s*float(R.get('relay_defocus_error_mm', 0.))*1e-3
        defocus = (self._n((B, max(o.passes-1, 0)), g)*focus_sigma
                   if o.geometry is not None and focus_sigma > 0 else None)
        disk_phase = disk_phase*o.block(self.disk_mask)
        hold = self.propagate_truth(dict(amp=amp, upstream=upstream, gain=gain, spatial=spatial,
                                         disk_phase=disk_phase, logamp=logamp, tilts=tilts,
                                         defocus=defocus), command)
        meas, var, valid, context = self.camera_frames(hold['planes'], fam, waist, prad, g)
        # Hold metrics are relative to the pump profile the network does not see.
        nominal_pump = o.pump_profile(prad)
        F, S, P = overlaps(hold['output'], desired[:, 0], hold['disk'], nominal_pump)
        truth_disk = torch.einsum('kij,bij->bk', o.disk_basis, o.disk_weight*disk_phase)
        theta = torch.cat([truth_slm, truth_disk, gain[:, None], gamma[:, None]], 1)
        batch = dict(amp=amp, command=command, target=target, meas=meas, var=var,
                     valid=valid, desired_planes=desired, context=context,
                     pump=nominal_pump, waist=waist, pump_radius=prad, family=fam)
        truth = dict(theta=theta, upstream=upstream, gain=gain, spatial=spatial,
                     disk_phase=disk_phase, logamp=logamp, tilts=tilts, defocus=defocus,
                     hold_fidelity=F,
                     hold_shape=S, hold_pump=P)
        return batch, truth

    def propagate_truth(self, truth, command, *, planes=True, quantize=True):
        o = self.optics
        phasor = o.render(command, truth['gain'], truth['spatial'], quantize=quantize)
        phasor = phasor*torch.polar(torch.ones_like(truth['upstream']), truth['upstream'])
        return o.forward(truth['amp'], phasor, None, truth['disk_phase'], truth['logamp'],
                         planes=planes, tilts=truth.get('tilts'), defocus=truth.get('defocus'))

    def trial(self, batch, index):
        """One synthetic sample as the same ``TrialInputs`` a measured trial gives."""
        get = lambda k: batch[k][index].detach().cpu().numpy()
        return TrialInputs(TARGETS[int(batch['family'][index])], float(batch['waist'][index]),
            float(batch['pump_radius'][index]), get('amp').astype(float),
            get('command').astype(float), get('target').astype(float), get('meas'),
            get('var'), get('valid').astype(bool), get('context'))

    def outcome(self, batch, truth, step, *, quantize=True):
        """True corrected fidelity/shape/pump overlap for an SLM step (batched)."""
        t = dict(truth, amp=batch['amp'])
        out = self.propagate_truth(t, batch['command']+step, planes=False, quantize=quantize)
        return overlaps(out['output'], batch['desired_planes'][:, 0], out['disk'], batch['pump'])

    def camera_frames(self, planes, fam, waist, pump_radius, g):
        o, cam, s = self.optics, self.camera, self.widen
        B = len(planes)
        intensity = planes.real**2+planes.imag**2
        # Camera misregistration and pulse pointing, in optical cells.
        shift = self._n((B, 2), g)*s*self.ranges['camera_shift_pixels']*self.pixel/o.dx
        fy = torch.fft.fftfreq(self.spec.opt_n, device=self.device, dtype=o.real)
        ramp = torch.exp(-2j*math.pi*(shift[:, 0, None, None]*fy[None, None, :] +
                                       shift[:, 1, None, None]*fy[None, :, None]))
        intensity = torch.clamp(torch.fft.ifft2(torch.fft.fft2(intensity)*ramp[:, None]).real, min=0)
        full = float(cam['full_well_e'])
        peak = (intensity*self.fov).amax((-3, -2, -1), keepdim=True)
        exposure = self._u(.25, .8, (B, 1, 1, 1), g)*full/torch.clamp(peak, min=1e-30)
        e = intensity*exposure
        bits = int(cam['adc_bits'])
        gain = full/(2**bits-1-float(cam['black_level_adu']))
        dark = float(cam['dark_current_e_s'])*float(cam['exposure_s'])+float(cam['background_e'])
        extra = float(cam['read_noise_e'])**2+float(cam['dsnu_rms_e'])**2+gain**2/12
        var = (e+dark+extra+(float(cam['prnu_rms'])*e)**2)/self.pixels_per_cell
        # Binned values include dark/background, as bin_camera's do.
        meas = torch.clamp(e+dark+var.sqrt()*self._n(e.shape, g), min=0)
        # A cell is lost when any pixel clips; the cell mean is below its peak.
        saturated = (e > .9*full)
        valid = (self.fov > 0) & ~saturated
        meas = torch.where(valid, meas, torch.zeros_like(meas))
        total = (meas*valid).sum((-2, -1))*self.pixels_per_cell
        sat_fraction = (saturated & (self.fov > 0)).sum((-2, -1))/self.fov.sum()
        fam_onehot = tnf.one_hot(fam, 4).to(o.real)
        context = torch.cat([fam_onehot, (10*(waist/self.spec.waist_m-1))[:, None],
                             (10*(pump_radius/self.spec.pump_radius_m-1))[:, None],
                             torch.log10(torch.clamp(total, min=1.))-8, 10*sat_fraction.to(o.real)], 1)
        return meas, var, valid, context


# ----------------------------------------------------------------------------
# Measurement-consistent refinement, planning and certification
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CertificateLimits:
    min_gain: float = .01
    min_shape: float = .95
    max_shape_drop: float = .012
    min_pump_ratio: float = .9     # disk-plane pump-overlap proxy for the 80% energy guard
    max_relative_residual: float = .35  # scale-free amplitude misfit (coarse screen)
    max_reduced_chi2: float = 3.        # noise-weighted misfit with model floor
    quantile: float = 0.           # 0 = worst posterior sample
    # Empirical per-target margin: certified = lower - base - slope*max(mean, 0).
    # Set by calibration against independent outcomes; defaults are placeholders.
    margin_base: tuple = (.03, .03, .03, .03)
    margin_slope: tuple = (.5, .5, .5, .5)
    shape_margin: tuple = (.02, .02, .02, .02)
    # Re-planned candidates (planned, planned_x*) get their own margins: they
    # overfit the fitted model, so sharing one margin blocked the network step.
    planned_margin_base: tuple = (.03, .03, .03, .03)
    planned_margin_slope: tuple = (.5, .5, .5, .5)
    planned_shape_margin: tuple = (.02, .02, .02, .02)
    calibrated: bool = False
    samples: int = 32
    covariance_inflation: float = 4.
    hypothesis_chi2: float = 25.   # competing minima within this chi2 stay in the posterior
    refine_iterations: int = 60
    plan_iterations: int = 40
    plan_samples: int = 8


class OneShotController:
    """Network proposal -> measured-state refinement -> robust plan -> certificate."""

    def __init__(self, spec: Spec, networks, device='cpu', limits=CertificateLimits(),
                 precision='double'):
        if not networks:
            raise ValueError('at least one trained network is required')
        self.spec, self.limits = spec, limits
        self.fast = Optics(spec, device, 'single')
        self.exact = Optics(spec, device, precision)
        self.networks = [n.to(self.fast.device).eval() for n in networks]
        self.scale = self.exact.tensor(STATE_SCALE)
        self.prior = self.exact.tensor(PRIOR_SIGMA)

    def _batch(self, optics, trial: TrialInputs):
        t = optics.tensor
        waist = t([trial.waist_m])
        target = t(trial.target)[None]
        return dict(amp=t(trial.amp)[None], command=t(trial.command)[None], target=target,
                    meas=t(trial.meas)[None], var=t(trial.var)[None],
                    valid=t(trial.valid.astype(np.float32))[None],
                    desired_planes=optics.desired(waist, target)['planes'],
                    context=t(trial.context)[None],
                    pump=optics.pump_profile(t([trial.pump_radius_m])))

    def propose(self, trial: TrialInputs, *, seed=0, allow_uncalibrated=False):
        """Return the selected command and its certificate.

        ``allow_uncalibrated`` releases a candidate using placeholder margins;
        use it only for offline solver replay or calibration, never hardware.
        """
        lim, ex = self.limits, self.exact
        fast_batch = self._batch(self.fast, trial)
        with torch.no_grad():
            images, context = network_inputs(self.fast, fast_batch)
            outs = [net(images, context) for net in self.networks]
        real = self.exact.real
        mean = torch.stack([o['state_mean'][0] for o in outs]).to(real)*self.scale.to(self.fast.device)
        nn_coeff = torch.stack([o['coefficients'][0] for o in outs]).mean(0).to(real)
        nn_res = torch.stack([o['residual'][0] for o in outs]).mean(0).to(real)
        b = self._batch(ex, trial)
        mean = mean.to(ex.device)
        starts = [*mean, torch.zeros(STATE, dtype=ex.real, device=ex.device)]
        hypotheses = [self._refine(b, s) for s in starts]
        hypotheses.sort(key=lambda h: h['loss'])
        # Always test the phase-retrieval twins of the best fit explicitly: the
        # SLM-plane and disk-plane even modes can flip together or separately.
        best_theta = hypotheses[0]['theta']
        for twin in TWINS:
            hypotheses.append(self._refine(b, best_theta*ex.tensor(twin)))
        hypotheses.sort(key=lambda h: h['loss'])
        best = hypotheses[0]
        retained = [h for h in hypotheses
                    if h['chi2']-best['chi2'] <= max(lim.hypothesis_chi2, .02*best['chi2'])]
        unique = []
        for h in retained:
            if all(torch.linalg.norm(h['theta']-u['theta']) > 1e-3 for u in unique):
                unique.append(h)
        gen = torch.Generator(device=ex.device).manual_seed(int(seed))
        samples = [best['theta'][None]]
        per = max(1, lim.samples//len(unique))
        for h in unique:
            cov = self._covariance(b, h)
            chol = torch.linalg.cholesky(cov+1e-12*torch.eye(STATE, dtype=ex.real, device=ex.device))
            z = torch.randn((per, STATE), generator=gen, dtype=ex.real, device=ex.device)
            samples.append(h['theta'][None]+z@chol.T)
            if h is not best:
                samples.append(h['theta'][None])
        samples = torch.cat(samples)
        nn_step = ex.step_map(nn_coeff.to(ex.device)[None], nn_res.to(ex.device)[None])[0]
        plan_step, plan_coeff, plan_res = self._plan(b, samples[:lim.plan_samples],
                                                     nn_coeff.to(ex.device), nn_res.to(ex.device))
        candidates = [('hold', torch.zeros_like(nn_step)), ('network', nn_step),
                      ('planned', plan_step)]
        candidates += [(f'planned_x{a:g}', a*plan_step) for a in (.75, .5, .25)]
        family = TARGETS.index(trial.family)
        scores = self._score(b, samples, candidates, family)
        chi2_reduced = best['chi2']/max(best['n']-STATE, 1)
        consistent = bool(best['relative'] <= lim.max_relative_residual and
                          chi2_reduced <= lim.max_reduced_chi2)
        decision, reason = self._decide(scores, consistent)
        if not (lim.calibrated or allow_uncalibrated):
            decision, reason = 0, 'hold_uncalibrated'
        name, step = candidates[decision]
        command = np.mod(trial.command+step.detach().cpu().numpy(), 2*np.pi)
        certificate = dict(schema=SCHEMA, decision=name, status=reason,
            applied=decision != 0,
            requires_measured_verification=decision != 0,
            model_consistent=consistent,
            relative_amplitude_residual=float(best['relative']),
            reduced_chi2=float(best['chi2']/max(best['n']-STATE, 1)),
            hypotheses=len(unique), posterior_samples=int(len(samples)),
            state_estimate=best['theta'].tolist(),
            network_state_estimate=mean.mean(0).tolist(),
            candidates={c: v for c, v in zip((c[0] for c in candidates), scores)},
            limits=asdict(lim),
            scope=('Certificate over a Laplace/multi-start posterior of a frozen passive '
                   'SLM/disk-screen model fitted to the two measured frames. Not a full '
                   'amplifier/thermal solver guarantee; verify after application.'))
        coefficients = plan_coeff if name.startswith('planned') else nn_coeff
        factor = float(name.split('_x')[1]) if '_x' in name else 1.
        return dict(command=command, step=step.detach().cpu().numpy(),
                    candidate_steps={c: v.detach().cpu().numpy() for c, v in candidates},
                    coefficients=(factor*coefficients).detach().cpu().numpy() if decision else np.zeros(SLM_MODES),
                    residual=(factor*(plan_res if name.startswith('planned') else nn_res)).detach().cpu().numpy()
                             if decision else np.zeros((RESIDUAL, RESIDUAL)),
                    certificate=certificate)

    # -- refinement ---------------------------------------------------------
    def _residual(self, b, theta):
        out = self.exact.model(theta[None], b['amp'], b['command'], b['pump'])
        planes = out['planes']
        return amplitude_fit(planes.real**2+planes.imag**2, b['meas'], b['var'], b['valid'])

    def _refine(self, b, start):
        theta = start.detach().clone().requires_grad_(True)
        n = float(b['valid'].sum())
        opt = torch.optim.LBFGS([theta], lr=1., max_iter=self.limits.refine_iterations,
                                history_size=20, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-10, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            _, chi2, _ = self._residual(b, theta)
            loss = (.5*chi2.sum()+.5*((theta/self.prior)**2).sum())/n
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            _, chi2, rel = self._residual(b, theta)
            loss = (.5*chi2.sum()+.5*((theta/self.prior)**2).sum())/n
        return dict(theta=theta.detach(), chi2=float(chi2), relative=float(rel),
                    loss=float(loss), n=n)

    def _covariance(self, b, h):
        valid = b['valid'].bool()

        def res(theta):
            r, _, _ = self._residual(b, theta)
            return r[valid]
        try:
            jac = torch.func.jacfwd(res)(h['theta'])
        except Exception:
            # Central differences: 60 smooth double-precision model calls.
            eye = torch.eye(STATE, dtype=h['theta'].dtype, device=h['theta'].device)
            steps = 1e-5*self.prior
            jac = torch.stack([(res(h['theta']+d*e)-res(h['theta']-d*e))/(2*d)
                               for d, e in zip(steps, eye)], 1)
        # Model-error inflation weakens the measured likelihood only; the
        # prior keeps unobservable directions at their physical range.
        dof = max(h['n']-STATE, 1.)
        inflate = max(1., h['chi2']/dof)*self.limits.covariance_inflation
        hess = jac.T@jac/inflate+torch.diag(1/self.prior**2)
        cov = torch.linalg.inv(hess)
        return .5*(cov+cov.T)

    # -- planning -----------------------------------------------------------
    def _metrics(self, b, samples, step, quantize=True):
        ex = self.exact
        k = len(samples)
        out = ex.model(samples, b['amp'].expand(k, -1, -1), b['command'].expand(k, -1, -1),
                       b['pump'].expand(k, -1, -1), planes=False, step=step, quantize=quantize)
        return overlaps(out['output'], b['desired_planes'][:, 0].expand(k, -1, -1),
                        out['disk'], b['pump'].expand(k, -1, -1))

    def _plan(self, b, samples, coeff0, res0):
        lim, ex = self.limits, self.exact
        with torch.no_grad():
            F0, S0, P0 = self._metrics(b, samples, None)
        shape_req = torch.clamp(S0-lim.max_shape_drop, min=lim.min_shape)
        best = None
        for c_init, r_init in ((coeff0, res0), (torch.zeros_like(coeff0), torch.zeros_like(res0))):
            c = c_init.detach().clone().requires_grad_(True)
            r = r_init.detach().clone().requires_grad_(True)
            opt = torch.optim.LBFGS([c, r], lr=1., max_iter=lim.plan_iterations, history_size=20,
                                    line_search_fn='strong_wolfe')

            def objective():
                step = ex.step_map(c[None], r[None])
                F, S, P = self._metrics(b, samples, step, quantize=False)
                return ((1-F).mean()+50*torch.relu(shape_req-S).square().mean()
                        + 50*torch.relu(lim.min_pump_ratio-P/P0).square().mean()
                        + 1e-3*r.square().mean())

            def closure():
                opt.zero_grad()
                loss = objective()
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                loss = float(objective())
            if best is None or loss < best[0]:
                best = (loss, c.detach(), r.detach())
        _, c, r = best
        return ex.step_map(c[None], r[None])[0].detach(), c, r

    # -- certification ------------------------------------------------------
    def _score(self, b, samples, candidates, family):
        lim = self.limits
        def margins(name):
            if name.startswith('planned'):
                return (lim.planned_margin_base[family], lim.planned_margin_slope[family],
                        lim.planned_shape_margin[family])
            return lim.margin_base[family], lim.margin_slope[family], lim.shape_margin[family]
        with torch.no_grad():
            F0, S0, P0 = self._metrics(b, samples, None)
            scores = []
            for name, step in candidates:
                base, slope, shape_margin = margins(name)
                F, S, P = self._metrics(b, samples, step[None])
                gain = F-F0
                q = float(torch.quantile(gain, lim.quantile)) if lim.quantile else float(gain.min())
                required = torch.clamp(S0-lim.max_shape_drop, min=lim.min_shape)
                slack = float((S-required).min())
                shape_ok = slack >= shape_margin
                pump_ok = bool(torch.all(P/P0 >= lim.min_pump_ratio))
                mean = float(gain.mean())
                certified = q-base-slope*max(mean, 0.) if name != 'hold' else 0.
                scores.append(dict(gain_certified=certified, gain_lower=q, gain_mean=mean,
                    fidelity_map=float(F[0]), fidelity_min=float(F.min()),
                    fidelity_hold_map=float(F0[0]), shape_min=float(S.min()),
                    pump_ratio_min=float((P/P0).min()), shape_slack=slack,
                    shape_ok=shape_ok, pump_ok=pump_ok))
        return scores

    def _decide(self, scores, consistent):
        if not consistent:
            return 0, 'hold_measurement_inconsistent_with_model'
        lim = self.limits
        best, value = 0, -np.inf
        for i, s in enumerate(scores[1:], 1):
            if (s['shape_ok'] and s['pump_ok'] and s['gain_certified'] >= lim.min_gain
                    and s['gain_certified'] > value):
                best, value = i, s['gain_certified']
        if not best:
            return 0, 'hold_no_certified_improvement'
        return best, 'certified_requires_measured_verification'


def fit_margins(rows, *, target_coverage=.95, min_rows=20, min_gain=.01, harm=-.005,
                slopes=(0., .25, .5, .75, 1., 1.5)):
    """Per-target empirical gain and shape margins from independent outcomes.

    ``rows``: dicts with family, gain_lower, gain_mean, shape_slack, pump_ok
    (predicted by the certificate) and true_gain, true_shape_slack, guard_ok
    (observed). Shape slack is the worst sampled shape minus its requirement.

    The shape margin is the ``target_coverage`` quantile of the predicted minus
    observed slack. For each gain slope the base is the same quantile of the
    needed shift, so certified gain <= true gain for that fraction of *all*
    candidates. The chosen pair maximises released true gain with no released
    harmful or guard-violating candidate. Targets with fewer than ``min_rows``
    rows stay uncalibrated, which makes the controller hold.
    """
    report, base_out, slope_out, shape_out = {}, [], [], []
    for family in TARGETS:
        sel = [r for r in rows if r['family'] == family]
        if len(sel) < min_rows:
            report[family] = dict(rows=len(sel), calibrated=False)
            base_out.append(1.)
            slope_out.append(1.)
            shape_out.append(1.)
            continue
        low = np.array([r['gain_lower'] for r in sel])
        mean = np.maximum(np.array([r['gain_mean'] for r in sel]), 0)
        true = np.array([r['true_gain'] for r in sel])
        guard = np.array([r['guard_ok'] for r in sel], bool)
        slack = np.array([r['shape_slack'] for r in sel])
        true_slack = np.array([r['true_shape_slack'] for r in sel])
        shape_margin = max(0., float(np.quantile(slack-true_slack, target_coverage, method='higher')))
        eligible = (slack >= shape_margin) & np.array([r['pump_ok'] for r in sel], bool)
        best = None
        for slope in slopes:
            need = low-slope*mean-true
            base = max(0., float(np.quantile(need, target_coverage, method='higher')))
            certified = low-base-slope*mean
            released = (certified >= min_gain) & eligible
            bad = released & ((true < harm) | ~guard)
            score = float(true[released].sum()) if not bad.any() else -np.inf
            if best is None or score > best[0]:
                best = (score, base, slope, float(np.mean(certified <= true)),
                        float(released.mean()), int(bad.sum()))
        score, base, slope, coverage, released, bad = best
        ok = np.isfinite(score)
        report[family] = dict(rows=len(sel), calibrated=bool(ok), base=base, slope=slope,
                              shape_margin=shape_margin, coverage=coverage,
                              shape_coverage=float(np.mean(slack-shape_margin <= true_slack)),
                              released_fraction=released, released_harmful=bad)
        base_out.append(base if ok else 1.)
        slope_out.append(slope if ok else 1.)
        shape_out.append(shape_margin if ok else 1.)
    return tuple(base_out), tuple(slope_out), tuple(shape_out), report


def load_controller(folder, device='cpu', *, precision='double', calibrated=True):
    """Trained ensemble + calibration from a model directory (``fit`` output).

    Returns (model_json, OneShotController). A calibration whose weight
    fingerprints differ from the loaded weights is refused.
    """
    import hashlib
    import json
    from pathlib import Path
    folder = Path(folder)
    spec_json = json.loads((folder/'model.json').read_text())
    if spec_json['schema'] != SCHEMA:
        raise ValueError('incompatible model schema')
    spec = Spec.from_json(spec_json['spec'])
    files = sorted(folder.glob('member_*/weights.pt'))
    if not files:
        raise ValueError(f'no trained network weights in {folder}')
    nets = []
    for member in files:
        net = OneShotNet(spec.crop)
        net.load_state_dict(torch.load(member, map_location='cpu'))
        nets.append(net)
    limits = CertificateLimits()
    if calibrated and (folder/'calibration.json').exists():
        calibration = json.loads((folder/'calibration.json').read_text())
        weights = {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        if calibration['weights_sha256'] != weights:
            raise ValueError('stale calibration after weight changes; recalibrate')
        values = calibration['limits']
        limits = CertificateLimits(**{**values, **{k: tuple(values[k]) for k in (
            'margin_base', 'margin_slope', 'shape_margin', 'planned_margin_base',
            'planned_margin_slope', 'planned_shape_margin') if k in values}})
    return spec_json, OneShotController(spec, nets, device, limits, precision=precision)
