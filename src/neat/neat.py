# src/neat/neat.py
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional

ACTIVATION_FUNCTIONS = {
    'tanh': math.tanh,
    'sigmoid': lambda x: 1.0 / (1.0 + math.exp(-x)),
    'relu': lambda x: max(0.0, x),
    'identity': lambda x: x,
}

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


class NodeGene:
    """A node (neuron) with its own activation function."""
    def __init__(self, node_id: int):
        self.id = node_id
        self.activation = 'tanh' # Default activation

    def copy(self):
        new_node = NodeGene(self.id)
        new_node.activation = self.activation
        return new_node

class NEATGenome:    
    def __init__(self, num_inputs: int, num_outputs: int, key: int = None):
        self.key = key if key is not None else get_new_genome_id()
        self.num_sensory_inputs = num_inputs
        self.bias_node_id = num_inputs # The bias node is the last input node
        # Total nodes = sensors + bias
        self.num_inputs = num_inputs + 1
        self.num_outputs = num_outputs
        self.genes: List[Gene] = []
        self.nodes: Dict[int, NodeGene] = {}
        self.fitness = None
        
        # Initialize nodes
        for i in range(self.num_inputs + self.num_outputs):
            self.nodes[i] = NodeGene(i)
        
        self._initialize_minimal_connections()

    def _initialize_minimal_connections(self):
        # Start with just bias connections to outputs
        for j in range(self.num_outputs):
            output_node_id = self.num_inputs + j
            weight = random.uniform(-2.0, 2.0)
            self.genes.append(Gene(self.bias_node_id, output_node_id, weight))
    
    def mutate(self, config: Dict):
        if random.random() < config.get('weight_mutation_rate', 0.8):
            self._mutate_weights(config)
        if random.random() < config.get('connection_add_rate', 0.05):
            self._mutate_add_connection()
        if random.random() < config.get('node_add_rate', 0.03):
            self._mutate_add_node()
        if random.random() < config.get('activation_mutate_rate', 0.03):
            self._mutate_activation()
        if random.random() < config.get('enable_disable_rate', 0.05):
            self._mutate_enable_disable()

    def _mutate_enable_disable(self):
        """Randomly toggles a gene's `enabled` status."""
        if self.genes:
            gene = random.choice(self.genes)
            gene.enabled = not gene.enabled

    def _mutate_weights(self, config):
        for gene in self.genes:
            if random.random() < config.get('weight_perturb_rate', 0.9):
                gene.weight += random.gauss(0, config.get('weight_mutation_power', 0.5))
            else:
                gene.weight = random.uniform(-2.0, 2.0)
            gene.weight = np.clip(gene.weight, -3.0, 3.0)
    
    def _mutate_add_connection(self):        
        for _ in range(20):
            in_node = random.choice(list(self.nodes.keys()))
            out_node = random.choice(list(self.nodes.keys()))

            if out_node < self.num_inputs:
                continue
            if any(g.in_node == in_node and g.out_node == out_node for g in self.genes):
                continue
            if self.creates_cycle(in_node, out_node):
                continue
            
            self.genes.append(Gene(in_node, out_node, random.uniform(-1.0, 1.0)))
            return # Exit after adding a connection
    
    def _mutate_add_node(self):
        enabled_genes = [g for g in self.genes if g.enabled]
        if not enabled_genes: return 
        
        gene = random.choice(enabled_genes)
        gene.enabled = False
        
        new_node_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        self.nodes[new_node_id] = NodeGene(new_node_id)
        
        self.genes.append(Gene(gene.in_node, new_node_id, 1.0))
        self.genes.append(Gene(new_node_id, gene.out_node, gene.weight))

    def _mutate_activation(self):
        """Randomly changes the activation function of a single non-input node."""
        mutable_nodes = [n for n in self.nodes.values() if n.id >= self.num_inputs]
        if not mutable_nodes: return
            
        node_to_mutate = random.choice(mutable_nodes)
        node_to_mutate.activation = random.choice(list(ACTIVATION_FUNCTIONS.keys()))


    def adapt_io(self, new_num_sensory_inputs: int, new_num_outputs: int):
        """
        Adapts the genome to a new number of inputs and outputs after a
        morphological mutation.
        """
        # Adapt inputs (if body grows)
        # The controller's padding/truncating handles any reduction
        if new_num_sensory_inputs > self.num_sensory_inputs:
            # Update counts and IDs
            self.num_sensory_inputs = new_num_sensory_inputs
            self.num_inputs = new_num_sensory_inputs + 1
            self.bias_node_id = new_num_sensory_inputs
            
            # Add new NodeGene objects for any new sensor or bias nodes
            for i in range(self.num_inputs):
                if i not in self.nodes:
                    self.nodes[i] = NodeGene(i)

        # Adapt outputs
        delta_out = new_num_outputs - self.num_outputs
        
        # If outputs were added to the body
        if delta_out > 0:
            for _ in range(delta_out):
                new_node_id = max(self.nodes.keys()) + 1 if self.nodes else 0
                self.nodes[new_node_id] = NodeGene(new_node_id)
                self.num_outputs += 1

                self.genes.append(Gene(self.bias_node_id, new_node_id, random.uniform(-1, 1)))

        # NOTE: For simplicity, I don't remove output nodes if the body shrinks
        # The controller method will simply ignore extra outputs from the network

    def copy(self):
        new_genome = NEATGenome(self.num_sensory_inputs, self.num_outputs, key=self.key)
        new_genome.genes = [g.copy() for g in self.genes]
        new_genome.nodes = {nid: n.copy() for nid, n in self.nodes.items()}
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
    else: # Handles p2 > p1 and p1 == p2
        better_parent, worse_parent = parent2, parent1
    
    child = NEATGenome(parent1.num_sensory_inputs, parent1.num_outputs)
    
    child.nodes = {nid: n.copy() for nid, n in better_parent.nodes.items()}
    child.genes = []
    
    better_genes = {g.innovation: g for g in better_parent.genes}
    worse_genes = {g.innovation: g for g in worse_parent.genes}
    
    for innovation in sorted(list(better_genes.keys())):
        better_gene = better_genes[innovation]
        worse_gene = worse_genes.get(innovation)
        
        if worse_gene is not None:
            new_gene = random.choice([better_gene, worse_gene]).copy()
            if not better_gene.enabled or not worse_gene.enabled:
                new_gene.enabled = random.random() > 0.75
        else:
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