import numpy as np
import pytest
from hoyag.thermal_views import cylindrical_temperature_derivatives


def test_linear_radial_axial_gradient_and_fourier_flux():
    r=np.array([0.,.1,.2,.4]);z=np.array([0.,.01,.02]);nphi=4
    rc=(r[:-1]+r[1:])/2;zc=(z[:-1]+z[1:])/2
    temp=300+2*rc[None,:,None]+3*zc[:,None,None]+np.zeros((2,3,nphi))
    result=cylindrical_temperature_derivatives(temp,r,z,14.)
    grad=result['grad_T_K_m'];flux=result['heat_flux_W_m2']
    np.testing.assert_allclose(grad[...,0],2.,rtol=0,atol=1e-11)
    np.testing.assert_allclose(grad[...,1],0.,rtol=0,atol=1e-11)
    np.testing.assert_allclose(grad[...,2],3.,rtol=0,atol=1e-11)
    np.testing.assert_allclose(flux[...,0],-28.,rtol=0,atol=1e-10)
    np.testing.assert_allclose(flux[...,2],-42.,rtol=0,atol=1e-10)


def test_unresolved_depth_is_explicitly_unavailable():
    with pytest.raises(ValueError,match='does not resolve'):
        cylindrical_temperature_derivatives(np.ones((1,3,4)),
            np.arange(4.),np.arange(2.),14.)
