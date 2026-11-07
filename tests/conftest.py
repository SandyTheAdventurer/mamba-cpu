import pytest

from mamba_cpu import _C


def _available_isas():
    isas = ["default"]
    if _C.set_cpu_isa("avx2"):
        isas.append("avx2")
    _C.set_cpu_isa("auto")
    return isas


@pytest.fixture(params=_available_isas())
def cpu_isa(request):
    """Runs a test once per kernel ISA this CPU supports."""
    assert _C.set_cpu_isa(request.param)
    yield request.param
    _C.set_cpu_isa("auto")
