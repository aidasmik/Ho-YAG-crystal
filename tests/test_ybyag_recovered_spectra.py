import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ybyag import material_data


ROOT = Path(__file__).resolve().parents[1] / 'research/ybyag'


def test_devido_reconstruction_reproduces_published_anchors():
    metadata = json.loads((ROOT/'devido2020/reconstruction.json').read_text())
    peak_300 = metadata['files']['300K_raw_data.txt']['valid_peak_sigma_abs_cm2']
    peak_100 = metadata['files']['100K_raw_data.txt']['valid_peak_sigma_abs_cm2']
    lower_80 = metadata['files']['80K_raw_data.txt']['largest_censored_lower_bound_cm2']
    assert .75e-20 < peak_300 < .95e-20
    assert 25e-20 < peak_100 < 31e-20
    assert 48e-20 < lower_80 < 51e-20
    with (ROOT/'devido2020/devido_zpl_reconstructed.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 7242
    assert any(row['quality_flag'] == 'isolated_high_spike' for row in rows)
    assert all(np.isnan(float(row['sigma_abs_usable_cm2']))
               for row in rows if row['quality_flag'] != 'valid')


def test_25at_hot_figure_is_separate_opt_in_data():
    with pytest.warns(material_data.ApproximationWarning):
        absorption, emission = material_data.hot_25at_figure_cross_sections_m2(
            np.array([941., 971., 1032.]), 300., allow_approximate=True)
    assert .65e-24 < absorption[0] < .8e-24
    assert 1.8e-24 < emission[2] < 2.1e-24
    with pytest.warns(material_data.ApproximationWarning):
        hot_abs, hot_em = material_data.hot_25at_figure_cross_sections_m2(
            np.array([941., 971., 1032.]), 450., allow_approximate=True)
    assert hot_abs[0] < absorption[0]
    assert hot_em[2] < emission[2]
    with pytest.raises(ValueError, match='25 at.%'):
        material_data.hot_25at_figure_cross_sections_m2(
            969., 350., yb_at_percent=15., allow_approximate=True)
    with pytest.raises(ValueError, match='allow_approximate'):
        material_data.hot_25at_figure_cross_sections_m2(969., 350.)
    with pytest.raises(ValueError, match='450'):
        material_data.hot_25at_figure_cross_sections_m2(
            969., 523.15, allow_approximate=True)


def test_tang_exact_concentration_room_temperature_absorption():
    values = []
    for concentration in (5, 10, 15):
        with pytest.warns(material_data.ApproximationWarning):
            values.append(material_data.tang2014_rt_absorption_coefficient_m1(
                941., yb_at_percent=concentration, allow_approximate=True))
    # Fig. 5 labels approximately 6, 11 and 16 cm^-1 near 941 nm.
    assert np.allclose(values, [600, 1100, 1600], atol=80)
    with pytest.raises(ValueError, match='5, 10 or 15'):
        material_data.tang2014_rt_absorption_coefficient_m1(
            941., yb_at_percent=12, allow_approximate=True)
    with pytest.raises(ValueError, match='allow_approximate'):
        material_data.tang2014_rt_absorption_coefficient_m1(
            941., yb_at_percent=10)
