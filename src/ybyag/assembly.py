"""YAG crystal properties on the shared generic copper/contact geometry."""
from copy import deepcopy
from . import material_data as data


def configuration(material, settings, mount):
    """Explicit constant-property engineering approximation near room temperature.

    Above 15 at.% the packaged CT conductivity fit is unavailable. Use the
    HT family's 300 K value with explicit resistivity interpolation in doping,
    held constant over 293.15–300 K. This is a named proxy, not extrapolated
    CT data or undoped conductivity disguised as a doped measurement.
    """
    cfg = deepcopy(mount)
    if settings.assembly_property_model != 'yag_rt_proxy':
        raise ValueError('Yb:YAG requires the yag_rt_proxy assembly, not LuAG properties')
    concentration = material.yb_at_percent
    family, reference = ('CT', 293.15) if concentration <= 15 else ('HT', 300.0)
    if abs(concentration - 15.0) < 1e-10:
        # Aggarwal supplies an actual 15 at.% table rather than requiring a
        # concentration interpolation. Interpolate only in temperature.
        conductivity = data.measured_thermal_conductivity_W_mK(reference, 15)
        thermal_mass = data.doped_RT_density_heat_capacity(15)
        conductivity_source = 'Aggarwal2005 measured/derived 15 at.% table; temperature interpolation to 293.15 K'
        thermal_mass_source = 'Aggarwal2005 measured 15 at.% RT density and volumetric heat capacity'
    else:
        conductivity = data.thermal_conductivity_doped(reference, concentration,
                                                       family=family, interpolate_doping=True)
        thermal_mass = {
            'rho_kg_m3': float(data.density_host_kg_m3(293.15)),
            'Cp_J_kgK': float(data.heat_capacity_host_J_kgK(293.15)),
        }
        conductivity_source = f'Cini2017 {family} at {reference:g} K; explicit doping-resistivity interpolation; held constant in near-RT assembly'
        thermal_mass_source = f'Sato2025 undoped YAG host proxy; not measured {concentration:g} at.% data'
    elastic = data.elastic_vrh_moduli()
    cfg['material'] = f'{concentration:g} at.% Yb:YAG; RT host/aggregate proxies'
    cfg['status'] = 'Generic copper/contact; YAG RT proxy; photoelasticity not applied'
    cfg['geometry'].update(disk_radius_m=settings.disk_radius_m,
                           disk_thickness_m=settings.thickness_m)
    cfg['thermal']['disk'] = {
        'conductivity_W_mK': float(conductivity),
        'density_kg_m3': thermal_mass['rho_kg_m3'],
        'heat_capacity_J_kgK': thermal_mass['Cp_J_kgK'],
    }
    cfg['mechanical']['disk'] = {
        'young_Pa': elastic['E_Pa'], 'poisson': elastic['nu'],
        'expansion_K1': float(data.thermal_expansion_per_K(293.15)),
        'stress_free_temperature_K': 293.15,
        'name': 'YAG host cubic tensor reduced to isotropic VRH; host expansion proxy',
    }
    cfg['optics'] = {
        'index': material.refractive_index(),
        'wavelength_m': material.signal_wavelength_nm*1e-9,
        'dn_dT_K1': float(data.dn_dT_per_K(293.15)),
        'reference_temperature_K': 293.15, 'photoelastic_model': None,
        'index_provenance': 'Zelmon1998 undoped YAG Sellmeier; doping correction unknown',
        'dn_dT_provenance': 'Aggarwal2005 host stress-free baseline at 1064 nm used as an explicit 1030 nm proxy',
    }
    mount_sources = {key: mount['sources'][key] for key in ('copper', 'interface', 'coolant', 'hardware')}
    # The inherited interface description names its original LuAG context;
    # neither material has a measured contact in this generic mount.
    mount_sources['interface'] = '15 kW/m²K generic indium/contact assumption; requires calibration for the actual YAG mount'
    cfg['sources'] = {
        **mount_sources,
        'selected_property_model': 'yag_rt_proxy',
        'dataset': 'Yb-YAG from GitHub origin/main 7aa9904',
        'conductivity': conductivity_source,
        'heat_capacity_density': thermal_mass_source,
        'elasticity': 'YAG host cubic tensor, isotropic Voigt-Reuss-Hill reduction',
        'expansion': 'Aggarwal2005 YAG host at 293.15 K',
        'thermooptic': cfg['optics']['dn_dT_provenance'],
        'photoelasticity': 'Not applied in scalar amplifier optics; tensor data exist but require oriented vector coupling',
    }
    return cfg
