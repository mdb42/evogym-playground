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
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.genes: List[Gene] = []
        self.nodes = set()
        self.fitness = None
        
        # Initialize nodes
        for i in range(num_inputs + num_outputs):
            self.nodes.add(i)
        
        self._initialize_minimal_connections()


    def _initialize_minimal_connections(self):
        # Connect each input to each output with high probability
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                if random.random() < 0.8:  # 80% chance to connect
                    out_node = self.num_inputs + j
                    weight = random.uniform(-2.0, 2.0)
                    self.genes.append(Gene(i, out_node, weight))
        
        # Ensure at least one connection per output
        for j in range(self.num_outputs):
            out_node = self.num_inputs + j
            has_connection = any(g.out_node == out_node for g in self.genes)
            if not has_connection:
                in_node = random.randint(0, self.num_inputs - 1)
                weight = random.uniform(-2.0, 2.0)
                self.genes.append(Gene(in_node, out_node, weight))
    
    def mutate(self, config: Dict):
        # Weight mutations
        if random.random() < config.get('weight_mutation_rate', 0.8):
            self._mutate_weights(config)
        
        # Structural mutations
        if random.random() < config.get('connection_add_rate', 0.05):
            self._mutate_add_connection()
        
        if random.random() < config.get('node_add_rate', 0.03):
            self._mutate_add_node()
    
    def _mutate_weights(self, config):
        for gene in self.genes:
            if random.random() < config.get('weight_perturb_rate', 0.9):
                # Perturb weight
                gene.weight += random.gauss(0, config.get('weight_mutation_power', 0.5))
                gene.weight = np.clip(gene.weight, -3.0, 3.0)
            else:
                # Replace weight
                gene.weight = random.uniform(-2.0, 2.0)
    
    def _mutate_add_connection(self):        
        # Try up to 20 times to find a valid new connection
        for _ in range(20):
            nodes_list = list(self.nodes)
            in_node = random.choice(nodes_list)
            out_node = random.choice(nodes_list)

            # Ensure the output node is not an input node
            if out_node < self.num_inputs:
                continue

            # Check if this connection already exists
            existing = any(g.in_node == in_node and g.out_node == out_node for g in self.genes)
            if existing:
                continue

            # Check if the new connection would create a cycle
            if self.creates_cycle(in_node, out_node):
                continue
            
            # If all checks pass, add the new gene
            weight = random.uniform(-1.0, 1.0)
            self.genes.append(Gene(in_node, out_node, weight))
            return # Exit after adding a connection
    
    def _mutate_add_node(self):
        if not self.genes:
            return
        
        enabled_genes = [g for g in self.genes if g.enabled]
        if not enabled_genes:
            return
            
        gene = random.choice(enabled_genes)
        gene.enabled = False
        
        new_node = max(self.nodes) + 1
        self.nodes.add(new_node)
        
        self.genes.append(Gene(gene.in_node, new_node, 1.0))
        self.genes.append(Gene(new_node, gene.out_node, gene.weight))

    def adapt_io(self, new_num_inputs: int, new_num_outputs: int):
        delta_out = new_num_outputs - self.num_outputs
        
        # Add new output nodes
        for _ in range(delta_out):
            new_node_id = max(self.nodes) + 1
            self.nodes.add(new_node_id)
            self.num_outputs += 1
            # Connect the new output node to a random existing node
            possible_in_nodes = [n for n in self.nodes if n < self.num_inputs or n >= self.num_inputs + self.num_outputs]
            if not possible_in_nodes: possible_in_nodes = list(range(self.num_inputs)) # Fallback
            in_node = random.choice(possible_in_nodes)
            self.genes.append(Gene(in_node, new_node_id, random.uniform(-1, 1)))
            
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
        new_genome = NEATGenome(self.num_inputs, self.num_outputs, key=self.key)
        new_genome.genes = [g.copy() for g in self.genes]
        new_genome.nodes = self.nodes.copy()
        new_genome.fitness = self.fitness
        return new_genome
    
    def creates_cycle(self, in_node: int, out_node: int) -> bool:
        if in_node == out_node:
            return True

        # Use DFS to find a path from out_node back to in_node
        visited = {out_node}
        stack = [out_node]
        while stack:
            current = stack.pop()
            if current == in_node:
                return True # Path exists, adding the connection would create a cycle.
            
            for gene in self.genes:
                if gene.enabled and gene.in_node == current:
                    neighbor = gene.out_node
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        
        return False # No path found

    @staticmethod
    def create_for_morphology(body):
        num_outputs = np.sum((body == 3) | (body == 4))
        num_inputs = 8
        return NEATGenome(num_inputs, num_outputs)