# src/neat/__init__.py
from .neat import NEATGenome, Gene, InnovationCounter
from .network import NEATNetwork
from .species import Species, SpeciesManager

__all__ = [
    'NEATGenome', 'Gene', 'InnovationCounter',
    'NEATNetwork', 
    'Species', 'SpeciesManager',
    'NEATEvolution',
    'crossover', 'crossover_with_mutation'
]