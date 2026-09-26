"""The optional native pump loop must preserve the reference photon transport."""

import numpy as np
import pytest

from ybluag.multipass_pump import transport_multipass_pump
from ybluag.native_pump import transport_fixed_temperature
from ybluag.model import YbLuAGMaterial
from ybyag.model import YbYAGMaterial


@pytest.mark.skipif(transport_fixed_temperature is None, reason="Numba extra missing")
@pytest.mark.parametrize("passes", [1, 2, 9, 10])
@pytest.mark.parametrize(
    "material", [YbYAGMaterial(yb_at_percent=10), YbLuAGMaterial()]
)
def test_compiled_pump_matches_numpy_transport(monkeypatch, passes, material):
    rng = np.random.default_rng(27)
    pump = rng.uniform(1e4, 2e6, (8, 9))
    scale = rng.uniform(0.7, 1.3, (2, 8, 9))
    beta = rng.uniform(0.02, 0.95, scale.shape)
    args = (material, pump, scale, 100e-6, passes, beta)

    monkeypatch.setenv("YB_PUMP_BACKEND", "python")
    reference = transport_multipass_pump(*args, relay_efficiency=0.97)
    monkeypatch.delenv("YB_PUMP_BACKEND")
    compiled = transport_multipass_pump(*args, relay_efficiency=0.97)

    for expected, actual in zip(reference, compiled):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-9)
    absorbed = float(np.sum(compiled[1]))
    assert np.isfinite(absorbed)
