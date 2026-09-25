"""Software/data-integrity tests. Passing is not experimental validation."""
import csv
import hashlib
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import yb_yag as m


class MaterialTests(unittest.TestCase):
    def test_source_hashes(self):
        expected={'lambda_a.txt':'8f7f44d73d83241dc0b952fd0a960072924c833f',
                  'lambda_e.txt':'8f7f44d73d83241dc0b952fd0a960072924c833f',
                  'sigma_a.txt':'a8f3582cf2475af0ee9323cce3a8a432a1395a55',
                  'sigma_e.txt':'64c2f75ca9a613a613568a450ba35f37c0c40bf5'}
        for name,sha in expected.items():
            b=(ROOT/'spectra/haseongpu_original'/name).read_bytes()
            self.assertEqual(hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),sha)

    def test_csv_matches_original(self):
        a=np.loadtxt(ROOT/'spectra/haseongpu_original/sigma_a.txt')
        e=np.loadtxt(ROOT/'spectra/haseongpu_original/sigma_e.txt')
        aa,ee=m.cross_sections_cm2(np.arange(905,1096))
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(e,ee)

    def test_spectral_anchors(self):
        a,_=m.cross_sections_cm2(940);_,e=m.cross_sections_cm2(1030)
        self.assertAlmostEqual(a/7.7796e-21,1,12)
        self.assertAlmostEqual(e/2.4778e-20,1,12)

    def test_spectral_nonnegative(self):
        for v in m.cross_sections_cm2(np.arange(905,1096)):
            self.assertTrue(np.all(np.isfinite(v)&(v>=0)))

    def test_units(self):
        a,e=m.cross_sections_cm2(940); aa,ee=m.cross_sections_m2(940)
        self.assertEqual(aa,a*1e-4);self.assertEqual(ee,e*1e-4)

    def test_rt_T_guard(self):
        with self.assertRaises(ValueError):m.cross_sections_cm2(940,300)

    def test_spectral_range_guard(self):
        for wl in [880,1100,np.nan,np.inf]:
            with self.assertRaises(ValueError):m.cross_sections_cm2(wl)

    def test_unknown_dataset(self):
        with self.assertRaises(ValueError):m.cross_sections_cm2(1030,dataset='unknown')

    def test_approximation_opt_in(self):
        with self.assertRaises(ValueError):m.cross_sections_cm2(1030,353.15,dataset='laser_band_figure')

    def test_approximate_broadcast(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a,e=m.cross_sections_cm2(np.array([1020,1030,1060])[None,:],np.array([293.15,373.15,473.15])[:,None],dataset='laser_band_figure',allow_approximate=True)
        self.assertEqual(a.shape,(3,3));self.assertEqual(e.shape,(3,3))
        self.assertTrue(np.all(a>0));self.assertTrue(np.all(e>0))

    def test_missing_pump_T_data_guard(self):
        with self.assertRaises(ValueError):m.cross_sections_cm2(940,353.15,dataset='laser_band_figure',allow_approximate=True)

    def test_mccumber_same_T(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a,e=m.cross_sections_cm2(1030,353.15,dataset='laser_band_figure',allow_approximate=True)
        self.assertAlmostEqual(e/a,m.mccumber_emission_absorption_ratio(1030,353.15),10)

    def test_partition_low_T(self):
        l,u=m.partition_functions(1.0)
        self.assertAlmostEqual(l,2);self.assertAlmostEqual(u,2)

    def test_number_density(self):
        self.assertTrue(1.38e28 < m.site_density_m3() < 1.40e28)
        self.assertAlmostEqual(m.yb_number_density_m3(10)/m.yb_number_density_m3(5),2)
        for c in [-1,101,np.nan]:
            with self.assertRaises(ValueError):m.yb_number_density_m3(c)

    def test_gain_ground_state(self):
        self.assertAlmostEqual(m.gain_coefficient_m1(940,5,0),-m.absorption_coefficient_m1(940,5))

    def test_gain_transparency(self):
        a,e=m.cross_sections_cm2(1030);b=a/(a+e)
        self.assertAlmostEqual(m.gain_coefficient_m1(1030,5,b),0,10)
        with self.assertRaises(ValueError):m.gain_coefficient_m1(1030,5,1.1)

    def test_saturation(self):
        a,e=m.cross_sections_m2(1030)
        expected=m.HC_J_M/(1030e-9*(a+e))
        self.assertAlmostEqual(m.saturation_fluence_J_m2(1030)/expected,1,12)
        self.assertAlmostEqual(m.saturation_intensity_W_m2(1030)*.00095/expected,1,12)
        with self.assertRaises(ValueError):m.saturation_intensity_W_m2(1030,lifetime_s=0)

    def test_quantum_defect(self):
        self.assertAlmostEqual(m.quantum_defect_fraction(940,1030),1-940/1030)
        with self.assertRaises(ValueError):m.quantum_defect_fraction(1030,940)

    def test_sellmeier(self):
        self.assertTrue(1.81<m.n_yag(1030)<1.83)
        self.assertTrue(m.n_yag(400)>m.n_yag(1030)>m.n_yag(5000))
        with self.assertRaises(ValueError):m.n_yag(399)

    def test_cini_exact_anchor(self):
        self.assertAlmostEqual(m.thermal_conductivity_doped(199.38,4,family='CT'),11.72)
        with self.assertRaises(ValueError):m.thermal_conductivity_doped(200,5,family='HT')

    def test_doping_interpolation_explicit(self):
        with self.assertRaises(ValueError):m.thermal_conductivity_doped(350,10,family='HT')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            k=m.thermal_conductivity_doped(350,10,family='HT',interpolate_doping=True)
        self.assertTrue(m.thermal_conductivity_doped(350,22.9,family='HT')<k<m.thermal_conductivity_doped(350,9.4,family='HT'))

    def test_measured_k(self):
        self.assertEqual(m.measured_thermal_conductivity_W_mK(298,2),8.6)
        self.assertEqual(m.measured_thermal_conductivity_W_mK(101,15),16.4)
        with self.assertRaises(ValueError):m.measured_thermal_conductivity_W_mK(150,5)

    def test_doped_RT_cp(self):
        p=m.doped_RT_density_heat_capacity(2)
        self.assertEqual(p['rho_kg_m3'],4600)
        self.assertAlmostEqual(p['Cp_J_kgK'],2.64e6/4600)

    def test_host_thermal_anchors(self):
        self.assertEqual(m.heat_capacity_host_J_kgK(300),604)
        self.assertAlmostEqual(m.thermal_conductivity_host(300),10.4,delta=.12)
        self.assertAlmostEqual(m.thermal_diffusivity_host_m2_s(300),3.80e-6,delta=.06e-6)
        self.assertLess(m.thermal_conductivity_host(400,'ceramic'),m.thermal_conductivity_host(400))

    def test_host_thermal_guard(self):
        for fun,x in [(m.thermal_conductivity_host,100),(m.heat_capacity_host_J_kgK,600)]:
            with self.assertRaises(ValueError):fun(x)

    def test_thermooptic_anchors(self):
        self.assertAlmostEqual(m.thermal_expansion_per_K(300)/1e-6,6.14,delta=.02)
        self.assertAlmostEqual(m.dn_dT_per_K(300)/1e-6,7.83,delta=.02)

    def test_apparent_dn_guard(self):
        with self.assertRaises(ValueError):m.dn_dT_per_K(400,dataset='sato2025')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d=m.dn_dT_per_K(400,dataset='sato2025',allow_apparent=True)
        self.assertAlmostEqual(d/1e-6,8.88,delta=.04)

    def test_integrated_thermal_index(self):
        self.assertEqual(m.thermal_index_change(293.15),0)
        a=m.thermal_index_change(280,260);b=m.thermal_index_change(260,280)
        self.assertAlmostEqual(a,-b,14)
        with self.assertRaises(ValueError):m.thermal_index_change(300,wavelength_nm=1030)

    def test_density_expansion(self):
        self.assertEqual(m.density_host_kg_m3(300),4552)
        self.assertGreater(m.density_host_kg_m3(200),m.density_host_kg_m3(400))

    def test_elastic_symmetries(self):
        C=m.elastic_tensor_Pa()
        np.testing.assert_array_equal(C,C.transpose(1,0,2,3))
        np.testing.assert_array_equal(C,C.transpose(2,3,0,1))
        e=np.array([[1,2,0],[2,-1,0],[0,0,3]])*1e-5
        self.assertGreater(np.einsum('ij,ijkl,kl',e,C,e),0)
        self.assertTrue(0<m.elastic_vrh_moduli()['nu']<.5)

    def test_tensor_rotation(self):
        C=m.elastic_tensor_Pa();np.testing.assert_allclose(m.rotate_rank4(C,np.eye(3)),C)
        with self.assertRaises(ValueError):m.rotate_rank4(C,2*np.eye(3))

    def test_photoelastic_hydrostatic(self):
        e=np.eye(3)*1e-4
        np.testing.assert_allclose(m.photoelastic_delta_B(e),np.eye(3)*(-.029+2*.0091)*1e-4)

    def test_photoelastic_shear(self):
        e=np.zeros((3,3));e[0,1]=e[1,0]=1e-4
        self.assertAlmostEqual(m.photoelastic_delta_B(e)[0,1],2*(-.0615)*1e-4)
        with self.assertRaises(ValueError):m.photoelastic_delta_B(e,tensor_set='johnson1967_incomplete')

    def test_photoelastic_symmetry_guard(self):
        e=np.zeros((3,3));e[0,1]=1e-3
        with self.assertRaises(ValueError):m.photoelastic_delta_B(e)

if __name__=='__main__': unittest.main()
