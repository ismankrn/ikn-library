"""Metaheuristic algorithms."""

from ikn_library.algorithms.abc import ArtificialBeeColony
from ikn_library.algorithms.aco import AntColonyOptimization
from ikn_library.algorithms.algorithm import Algorithm
from ikn_library.algorithms.bat import BatAlgorithm
from ikn_library.algorithms.bees import BeesAlgorithm
from ikn_library.algorithms.binary_aco import BinaryAntColonyOptimization
from ikn_library.algorithms.camel import CamelAlgorithm
from ikn_library.algorithms.cso import CatSwarmOptimization
from ikn_library.algorithms.cuckoo import CuckooSearch
from ikn_library.algorithms.ga import GeneticAlgorithm
from ikn_library.algorithms.kma import KomodoMlipirAlgorithm
from ikn_library.algorithms.nsga2 import NSGA2
from ikn_library.algorithms.sa import SimulatedAnnealing

__all__ = [
    "NSGA2",
    "Algorithm",
    "AntColonyOptimization",
    "ArtificialBeeColony",
    "BatAlgorithm",
    "BeesAlgorithm",
    "BinaryAntColonyOptimization",
    "CamelAlgorithm",
    "CatSwarmOptimization",
    "CuckooSearch",
    "GeneticAlgorithm",
    "KomodoMlipirAlgorithm",
    "SimulatedAnnealing",
]
