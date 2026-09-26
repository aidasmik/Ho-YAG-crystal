"""Camera image with measured horizontal and vertical marginal beam profiles."""
from __future__ import annotations

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_camera_profiles(figure, slot, image, title, *, fov_width_mm,
                        baseline=0., unit="ADU", cmap="gray", vmax=None,
                        compact=False, display_stride=1, reference_image=None,
                        profiles=None, reference_profiles=None):
    """Plot the image and mean signal across each camera axis in physical mm.

    The top curve averages rows (x profile); the right curve averages columns
    (y profile). Camera black level is subtracted before forming ADU profiles.
    """
    data=np.asarray(image,float)
    if data.ndim!=2 or min(data.shape)<2 or not np.all(np.isfinite(data)):
        raise ValueError("camera profile needs a finite 2D image")
    if not np.isfinite(fov_width_mm) or fov_width_mm<=0:
        raise ValueError("camera field width must be positive")
    if not isinstance(display_stride,int) or display_stride<1:
        raise ValueError("display stride must be a positive integer")
    h,w=data.shape
    height_mm=fov_width_mm*h/w
    signal=np.maximum(data-baseline,0.)
    x_profile,y_profile=(signal.mean(axis=0),signal.mean(axis=1)) if profiles is None else (
        np.asarray(profiles[0],float),np.asarray(profiles[1],float))
    if (x_profile.ndim!=1 or y_profile.ndim!=1 or
            not np.all(np.isfinite(x_profile)) or not np.all(np.isfinite(y_profile))):
        raise ValueError("camera profiles must be finite 1D arrays")
    profile_x=(np.arange(len(x_profile))+.5-len(x_profile)/2)*fov_width_mm/len(x_profile)
    profile_y=(np.arange(len(y_profile))+.5-len(y_profile)/2)*height_mm/len(y_profile)
    if reference_image is not None:
        reference=np.asarray(reference_image,float)
        if reference.shape!=data.shape or not np.all(np.isfinite(reference)):
            raise ValueError("reference camera image must match the displayed image")
        reference_signal=np.maximum(reference-baseline,0.)
        reference_x,reference_y=(reference_signal.mean(axis=0),
            reference_signal.mean(axis=1)) if reference_profiles is None else (
            np.asarray(reference_profiles[0],float),
            np.asarray(reference_profiles[1],float))
        if reference_x.shape!=x_profile.shape or reference_y.shape!=y_profile.shape:
            raise ValueError("target profiles must match measured profile lengths")
    camera=figure.add_subplot(slot)
    divider=make_axes_locatable(camera)
    top=divider.append_axes("top",size="24%",pad=.08,sharex=camera)
    side=divider.append_axes("right",size="24%",pad=.08,sharey=camera)
    top.plot(profile_x,x_profile,color="#116d83",lw=1.15)
    if reference_image is not None:
        top.plot(profile_x,reference_x,color="#c66335",lw=1.,ls="--")
    top.set_ylim(0,max(float(x_profile.max())*1.08,
                       float(reference_x.max())*1.08 if reference_image is not None else 0.,1e-12))
    camera.imshow(data[::display_stride,::display_stride],origin="lower",cmap=cmap,
                  vmin=baseline if baseline else 0,
                  vmax=vmax,interpolation="nearest",
                  extent=(-fov_width_mm/2,fov_width_mm/2,-height_mm/2,height_mm/2))
    side.plot(y_profile,profile_y,color="#116d83",lw=1.15)
    if reference_image is not None:
        side.plot(reference_y,profile_y,color="#c66335",lw=1.,ls="--")
    side.set_xlim(0,max(float(y_profile.max())*1.08,
                        float(reference_y.max())*1.08 if reference_image is not None else 0.,1e-12))
    top.tick_params(axis="x",labelbottom=False)
    side.tick_params(axis="y",labelleft=False)
    top.set_title(title,fontsize=9 if compact else 11)
    camera.set_xlabel("x (mm)",fontsize=8)
    camera.set_ylabel("y (mm)",fontsize=8)
    if compact:
        top.set_yticks([])
        side.set_xticks([])
        camera.tick_params(labelsize=7)
    else:
        top.set_ylabel(f"mean {unit}",fontsize=8)
        side.set_xlabel(f"mean {unit}",fontsize=8)
        top.tick_params(labelsize=8)
        side.tick_params(labelsize=8)
        camera.tick_params(labelsize=8)
        side.set_xticks([side.get_xlim()[1]])
    return top,camera,side
