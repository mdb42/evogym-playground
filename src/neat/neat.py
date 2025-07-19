# src/neat/neat.py
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

_genome_id_counter = 0
def get_new_genome_id():
    global _genome_id_counter
    _genome_id_counter += 1
    return _genome_id_counter

class InnovationCounter:
    def __init__(self):
        self.innovations = {}  # (in_node, out_node) -> innovation_number
        self.current = 0
    
    def get_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self.innovations:
            self.innovations[key] = self.current
            self.current += 1
        return self.innovations[key]

innovation_counter = InnovationCounter()

class Gene:
    def __init__(self, in_node: int, out_node: int, weight: float, 
                 enabled: bool = True, innovation: int = None):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation or innovation_counter.get_innovation(in_node, out_node)
    
    def copy(self):
        return Gene(self.in_node, self.out_node, self.weight, 
                   self.enabled, self.innovation)

class NEATGenome:    
    def __init__(self, num_inputs: int, num_outputs: int, key: int = None):
        self.key = key if key is not None else get_new_genome_id()
        self.num_sensory_inputs = num_inputs
        self.bias_node_id = num_inputs # The bias node is the last input node
        # Total nodes = sensors + bias
        self.num_inputs = num_inputs + 1

        self.num_outputs = num_outputs
        self.genes: List[Gene] = []
        self.nodes = set()
        self.fitness = None
        
        # Initialize nodes
        for i in range(self.num_inputs + self.num_outputs):
            self.nodes.add(i)
        
        self._initialize_minimal_connections()

    def _initialize_minimal_connections(self):
        # Connect each input to each output with high probability
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                if random.random() < 0.8:  # 80% chance to connect
                    output_node_id = self.num_inputs + j
                    weight = random.uniform(-2.0, 2.0)
                    self.genes.append(Gene(i, output_node_id, weight))
        
        # Ensure at least one connection per output
        for j in range(self.num_outputs):
            output_node_id = self.num_inputs + j
            has_connection = any(g.out_node == output_node_id for g in self.genes)
            if not has_connection:
                random_input_id = random.randint(0, self.num_inputs - 1)
                weight = random.uniform(-2.0, 2.0)
                self.genes.append(Gene(random_input_id, output_node_id, weight))
    
    def mutate(self, config: Dict):
        if random.random() < config.get('weight_mutation_rate', 0.8):
            self._mutate_weights(config)
        if random.random() < config.get('connection_add_rate', 0.05):
            self._mutate_add_connection()
        if random.random() < config.get('node_add_rate', 0.03):
            self._mutate_add_node()

    def _mutate_weights(self, config):
        for gene in self.genes:
            if random.random() < config.get('weight_perturb_rate', 0.9):
                gene.weight += random.gauss(0, config.get('weight_mutation_power', 0.5))
            else:
                gene.weight = random.uniform(-2.0, 2.0)
            gene.weight = np.clip(gene.weight, -3.0, 3.0)
    
    def _mutate_add_connection(self):        
        for _ in range(20):
            in_node = random.choice(list(self.nodes))
            out_node = random.choice(list(self.nodes))

            if out_node < self.num_inputs:
                continue
            if any(g.in_node == in_node and g.out_node == out_node for g in self.genes):
                continue
            if self.creates_cycle(in_node, out_node):
                continue
            
            # If all checks pass, add the new gene
            weight = random.uniform(-1.0, 1.0)
            self.genes.append(Gene(in_node, out_node, weight))
            return # Exit after adding a connection
    
    def _mutate_add_node(self):
        enabled_genes = [g for g in self.genes if g.enabled]
        if not enabled_genes:
            return
        gene = random.choice(enabled_genes)
        gene.enabled = False
        
        new_node = max(self.nodes) + 1 if self.nodes else 0
        self.nodes.add(new_node)
        
        self.genes.append(Gene(gene.in_node, new_node, 1.0))
        self.genes.append(Gene(new_node, gene.out_node, gene.weight))

    def adapt_io(self, new_num_sensory_inputs: int, new_num_outputs: int):
        # Adapt inputs (if body grows)
        delta_in = new_num_sensory_inputs - self.num_sensory_inputs
        if delta_in > 0:
            for i in range(self.num_sensory_inputs, new_num_sensory_inputs):
                self.nodes.add(i)
            self.num_sensory_inputs = new_num_sensory_inputs
            self.num_inputs = new_num_sensory_inputs + 1
            self.bias_node_id = new_num_sensory_inputs

        # Adapt outputs
        delta_out = new_num_outputs - self.num_outputs

        for _ in range(delta_out):
            new_node_id = max(self.nodes) + 1 if self.nodes else 0
            self.nodes.add(new_node_id)
            self.num_outputs += 1
            # Connect the new output to the bias and a few random sensors
            self.genes.append(Gene(self.bias_node_id, new_node_id, random.uniform(-1, 1)))
            for _ in range(2): # Add two random sensory connections
                random_sensor = random.randint(0, self.num_sensory_inputs - 1)
                self.genes.append(Gene(random_sensor, new_node_id, random.uniform(-1, 1)))

        # Remove output nodes
        for _ in range(-delta_out):
            if self.num_outputs <= 1: break # Don't remove the last output
            # Find the last output node ID
            last_output_node_id = self.num_inputs + self.num_outputs - 1
            if last_output_node_id in self.nodes:
                self.nodes.remove(last_output_node_id)
            # Remove connections to this node
            self.genes = [g for g in self.genes if g.out_node != last_output_node_id]
            self.num_outputs -= 1

    def copy(self):
        new_genome = NEATGenome(self.num_sensory_inputs, self.num_outputs, key=self.key)
        new_genome.genes = [g.copy() for g in self.genes]
        new_genome.nodes = self.nodes.copy()
        new_genome.fitness = self.fitness
        return new_genome
    
    def creates_cycle(self, in_node: int, out_node: int) -> bool:
        if in_node == out_node: return True
        visited = {out_node}
        stack = [out_node]
        while stack:
            current = stack.pop()
            if current == in_node:
                return True # Path exists, adding the connection would create a cycle.
            for gene in self.genes:
                if gene.enabled and gene.in_node == current:
                    if gene.out_node not in visited:
                        visited.add(gene.out_node)
                        stack.append(gene.out_node)
        return False # No path found

    @staticmethod
    def create_for_morphology(body):
        num_sensory_inputs = np.sum(body > 0) * 2
        num_outputs = np.sum((body == 3) | (body == 4))
        return NEATGenome(num_sensory_inputs, num_outputs)


def crossover(parent1: NEATGenome, parent2: NEATGenome) -> NEATGenome:
    p1_fitness = parent1.fitness if parent1.fitness is not None else -float('inf')
    p2_fitness = parent2.fitness if parent2.fitness is not None else -float('inf')

    if p1_fitness > p2_fitness:
        better_parent, worse_parent = parent1, parent2
    elif p2_fitness > p1_fitness:
        better_parent, worse_parent = parent2, parent1
    else:
        # Equal fitness, randomly choose
        better_parent, worse_parent = random.choice([(parent1, parent2), (parent2, parent1)])
    
    child = NEATGenome(parent1.num_sensory_inputs, parent1.num_outputs)
    
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