"""Interaction datasets: drug-target interaction (DTI) benchmarks."""

from ikn_library.interactions.dti import DTIDataset, load_davis, load_kiba
from ikn_library.interactions.yamanishi import YamanishiDataset, load_yamanishi

__all__ = [
    "DTIDataset",
    "YamanishiDataset",
    "load_davis",
    "load_kiba",
    "load_yamanishi",
]
