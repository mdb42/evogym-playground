# src/neat/species.py
import numpy as np
import random
from typing import List, Dict, Optional
from .neat import NEATGenome

class Species:    
    def __init__(self, species_id: int, representative: NEATGenome):
        self.id = species_id
        self.representative = representative
        self.members: List[NEATGenome] = []
        self.fitness_history: List[float] = []
        self.age = 0
        self.last_improvement = 0
        
    def add_member(self, genome: NEATGenome):
        self.members.append(genome)
    
    def update(self):
        self.age += 1
        
        if self.members:
            # Calculate average fitness
            valid_members = [m for m in self.members if m.fitness is not None]
            if not valid_members:
                self.fitness_history.append(0.0)
                return

            avg_fitness = np.mean([m.fitness for m in valid_members])
            self.fitness_history.append(avg_fitness)
            
            # Track improvement
            if len(self.fitness_history) > 1:
                # Use max of previous history
                if avg_fitness > max(self.fitness_history[:-1] or [0]):
                    self.last_improvement = 0
                else:
                    self.last_improvement += 1
            
            # Select new representative randomly from members
            self.representative = random.choice(valid_members)
    
    def get_adjusted_fitness(self) -> float:
        valid_members = [m for m in self.members if m.fitness is not None]
        if not valid_members:
            return 0.0
        
        total = sum(m.fitness for m in valid_members)
        return total / len(valid_members)
    
    def is_stagnant(self, threshold: int) -> bool:
        return self.last_improvement > threshold

class SpeciesManager:    
    def __init__(self, config: Dict):
        self.config = config
        self.species: Dict[int, Species] = {}
        self.next_species_id = 0
        self.generation = 0
        
        self.c1 = config.get('excess_coefficient', 1.0)
        self.c2 = config.get('disjoint_coefficient', 1.0)
        self.c3 = config.get('weight_coefficient', 0.4)
        self.threshold = config.get('compatibility_threshold', 3.0)
    
    def speciate(self, genomes: List[NEATGenome]):
        # Clear existing members but keep the species object for history
        for s in self.species.values():
            s.members = []
        
        # Assign each genome to a species
        for genome in genomes:
            assigned = False
            for s in self.species.values():
                if self._compatibility_distance(genome, s.representative) < self.threshold:
                    s.add_member(genome)
                    assigned = True
                    break
            
            if not assigned:
                new_id = self.next_species_id
                self.next_species_id += 1
                self.species[new_id] = Species(new_id, genome)
                self.species[new_id].add_member(genome)
        
        # Remove empty species from active consideration
        self.species = {sid: s for sid, s in self.species.items() if s.members}
        
        # Update all species
        for s in self.species.values():
            s.update()
        
        self.generation += 1
        
        self._adjust_threshold()
    
    def _compatibility_distance(self, g1: NEATGenome, g2: NEATGenome) -> float:
        innovations1 = {gene.innovation for gene in g1.genes}
        innovations2 = {gene.innovation for gene in g2.genes}
        
        matching = innovations1.intersection(innovations2)
        
        max_innov_g1 = max(innovations1) if innovations1 else 0
        max_innov_g2 = max(innovations2) if innovations2 else 0
        
        excess = 0
        disjoint = 0
        
        # Count disjoint and excess genes
        for innov in innovations1:
            if innov not in innovations2:
                if innov > max_innov_g2:
                    excess += 1
                else:
                    disjoint += 1
        for innov in innovations2:
            if innov not in innovations1:
                if innov > max_innov_g1:
                    excess += 1
                else:
                    disjoint += 1
        
        # Calculate average weight difference for matching genes
        weight_diff = 0.0
        if matching:
            g2_genes_map = {g.innovation: g for g in g2.genes}
            for g1_gene in g1.genes:
                if g1_gene.innovation in g2_genes_map:
                    weight_diff += abs(g1_gene.weight - g2_genes_map[g1_gene.innovation].weight)
            weight_diff /= len(matching)
        
        # Normalize by larger genome size
        n = max(len(g1.genes), len(g2.genes))
        if n < 20: n = 1
        
        distance = (self.c1 * excess / n) + (self.c2 * disjoint / n) + (self.c3 * weight_diff)
        return distance
    
    def _adjust_threshold(self):
        target_species = self.config.get('target_species', 5)
        num_species = len(self.species)
        
        if num_species == 0: return

        if num_species < target_species:
            self.threshold *= 0.95
        elif num_species > target_species:
            self.threshold *= 1.05
        
        self.threshold = np.clip(self.threshold, 0.5, 10.0)

    def cull_stagnant_species(self):
        stagnation_threshold = self.config.get('stagnation_threshold', 15)

        if len(self.species) <= 2:
            return
        
        self.species = {
            sid: s for sid, s in self.species.items()
            if not s.is_stagnant(stagnation_threshold) or s.age < 5
        }

    def select_species_for_reproduction(self) -> Optional[Species]:
        active_species = list(self.species.values())
        if not active_species:
            return None

        adj_fitnesses = [s.get_adjusted_fitness() for s in active_species]

        # Shift fitness values to be non-negative for weighting
        min_adj_fitness = min(adj_fitnesses) if adj_fitnesses else 0
        weights = [(f - min_adj_fitness) for f in adj_fitnesses]
        
        total_weight = sum(weights)

        if total_weight <= 0:
            return random.choice(active_species)
        
        return random.choices(active_species, weights=weights, k=1)[0]
    
    def select_member(self, member_pool: List[NEATGenome]) -> Optional[NEATGenome]:
        if not member_pool:
            return None
        
        fitnesses = [m.fitness if m.fitness is not None else 0 for m in member_pool]
        
        # Normalize fitness to be non-negative for weighting
        min_fitness = min(fitnesses) if fitnesses else 0
        weights = [(f - min_fitness) for f in fitnesses]
        total_weight = sum(weights)

        if total_weight <= 0:
            return random.choice(member_pool)

        return random.choices(member_pool, weights=weights, k=1)[0]