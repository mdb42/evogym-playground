# src/neat/__init__.py
from .neat import NEATGenome, Gene, InnovationCounter
from .network import NEATNetwork
from .species import Species, SpeciesManager
from .neat_evolution import NEATEvolution
from .crossover import crossover, crossover_with_mutation

__all__ = [
    'NEATGenome', 'Gene', 'InnovationCounter',
    'NEATNetwork', 
    'Species', 'SpeciesManager',
    'NEATEvolution',
    'crossover', 'crossover_with_mutation'
]