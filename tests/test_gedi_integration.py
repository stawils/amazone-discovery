import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

h5py = pytest.importorskip("h5py", reason="h5py required for GEDI provider")

from src.providers.gedi_provider import GEDIProvider
from src.pipeline.modular_pipeline import ModularPipeline


def test_gedi_provider_creation():
    provider = GEDIProvider()
    assert provider is not None


def test_modular_pipeline_gedi(monkeypatch):
    def mock_search(self, zone, max_results=10):
        return []

    monkeypatch.setattr(GEDIProvider, "search_gedi_data", mock_search)
    pipeline = ModularPipeline(provider="gedi")
    results = pipeline.run(zones=["negro_madeira"], max_scenes=1)
    assert isinstance(results, dict)
