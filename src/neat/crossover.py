# src/neat/crossover.py
import random
from typing import Tuple, Optional
from .neat import NEATGenome, Gene

def crossover(parent1: NEATGenome, parent2: NEATGenome) -> NEATGenome:
    # Determine which parent is fitter
    if parent1.fitness is None and parent2.fitness is None:
        # No fitness info, randomly choose
        better_parent, worse_parent = random.choice([(parent1, parent2), (parent2, parent1)])
    elif parent1.fitness is None:
        better_parent, worse_parent = parent2, parent1
    elif parent2.fitness is None:
        better_parent, worse_parent = parent1, parent2
    elif parent1.fitness > parent2.fitness:
        better_parent, worse_parent = parent1, parent2
    elif parent2.fitness > parent1.fitness:
        better_parent, worse_parent = parent2, parent1
    else:
        # Equal fitness, randomly choose
        better_parent, worse_parent = random.choice([(parent1, parent2), (parent2, parent1)])
    
    # Create child genome with same I/O as parents
    child = NEATGenome(parent1.num_inputs, parent1.num_outputs)
    child.nodes = better_parent.nodes.copy()
    child.genes = []
    
    # Preserve body information from better parent
    if hasattr(better_parent, '_body'):
        child._body = better_parent._body
        child._connections = better_parent._connections
    
    # Create lookup dictionaries for quick access
    better_genes = {g.innovation: g for g in better_parent.genes}
    worse_genes = {g.innovation: g for g in worse_parent.genes}
    
    # All innovations from better parent
    all_innovations = set(better_genes.keys())
    
    # Process each innovation
    for innovation in sorted(all_innovations):
        better_gene = better_genes[innovation]
        
        if innovation in worse_genes:
            # Matching gene - randomly inherit from either parent
            worse_gene = worse_genes[innovation]
            
            if random.random() < 0.5:
                new_gene = better_gene.copy()
            else:
                new_gene = worse_gene.copy()
                
            # Inherit disabled statuses
            if not better_gene.enabled or not worse_gene.enabled:
                # 75% chance to be disabled if either parent has it disabled
                new_gene.enabled = random.random() > 0.75
        else:
            # Disjoint/excess gene
            new_gene = better_gene.copy()
        
        child.genes.append(new_gene)
    
    return child

def crossover_with_mutation(parent1: NEATGenome, parent2: NEATGenome, 
                           config: dict) -> NEATGenome:
    child = crossover(parent1, parent2)
    child.mutate(config)
    return child

def is_same_species(genome1: NEATGenome, genome2: NEATGenome, 
                   species_manager) -> bool:
    distance = species_manager._compatibility_distance(genome1, genome2)
    return distance < species_manager.threshold