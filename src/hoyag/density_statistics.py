"""Construct bounded concentration fluctuations without losing the mean.

Small perturbations retain the original affine Gaussian map. When that map
violates the requested floor, use a smooth monotone lognormal transform and
solve its variance. Thus the histogram may change; achieved statistics do not.
"""
import numpy as np


def bounded_mean_rms(smooth, mean_density_m3, relative_rms, minimum_fraction):
    s = np.asarray(smooth, float)
    for name,value in [('mean_density_m3',mean_density_m3),('relative_rms',relative_rms),('minimum_fraction',minimum_fraction)]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if minimum_fraction > 1:
        raise ValueError('a floor above the requested mean is infeasible')
    if not np.all(np.isfinite(s)) or s.size < 2:
        raise ValueError('finite nonconstant noise with at least two voxels required')
    if mean_density_m3 == 0:
        if relative_rms != 0:
            raise ValueError('relative RMS is undefined for a zero mean; request RMS=0')
        return np.zeros_like(s)
    if relative_rms == 0:
        return np.full_like(s, mean_density_m3)
    if minimum_fraction == 1:
        raise ValueError('positive RMS is incompatible with floor equal to the mean')
    s = s-s.mean()
    std = float(s.std())
    if std <= 0:
        raise ValueError('constant noise cannot represent nonzero RMS')
    s = s/std
    linear = 1+relative_rms*s
    if linear.min() >= minimum_fraction:
        return mean_density_m3*linear
    tied = int(np.count_nonzero(s == s.max()))
    largest = (1-minimum_fraction)*np.sqrt(s.size/tied-1)
    if relative_rms >= largest:
        raise ValueError(f'requested RMS is not attainable on this finite map (limit {largest:g})')
    def transform(scale):
        with np.errstate(under='ignore'):
            excess = np.exp(scale*(s-s.max()))
        excess = excess/excess.mean()
        return minimum_fraction+(1-minimum_fraction)*excess
    low, high = 0., 1.
    while transform(high).std() < relative_rms:
        high *= 2
        if not np.isfinite(high):
            raise ValueError('failed to bracket requested concentration variance')
    for _ in range(100):
        middle = (low+high)/2
        x = transform(middle)
        if x.std() < relative_rms:
            low = middle
        else:
            high = middle
    x = transform((low+high)/2)
    if not np.isclose(x.mean(),1.,rtol=2e-12,atol=0) or not np.isclose(x.std(),relative_rms,rtol=2e-10,atol=1e-14) or x.min() < minimum_fraction:
        raise FloatingPointError('concentration mean/RMS/floor constraints not satisfied')
    return mean_density_m3*x


def density_statistics(values_m3):
    a=np.asarray(values_m3,float)
    if np.any(~np.isfinite(a)) or np.any(a<0) or a.size==0:
        raise ValueError('density must be nonempty, finite and nonnegative')
    mean=float(a.mean())
    return {'mean_density_m3':mean,'minimum_density_m3':float(a.min()),
            'maximum_density_m3':float(a.max()),
            'relative_rms':None if mean==0 else float(a.std()/mean)}
