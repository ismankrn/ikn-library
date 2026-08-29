"""Microarray data processing: load GEO datasets into ML-ready tables."""

from ikn_library.microarray.geo import GEODataset, load_geo
from ikn_library.microarray.preprocess import top_variance

__all__ = ["GEODataset", "load_geo", "top_variance"]
