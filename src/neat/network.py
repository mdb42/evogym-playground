# src/neat/network.py
import numpy as np
import math
from typing import Dict, List
from .neat import NEATGenome, Gene

class NEATNetwork:    
    def __init__(self, genome: NEATGenome):
        self.genome = genome
        self.nodes: Dict[int, float] = {}
        self.node_layers: Dict[int, int] = {}
        self._build_layers()

    def _build_layers(self):
        """Builds the network layers for feed-forward activation."""
        # Input nodes (sensors + bias) are layer 0
        for i in range(self.genome.num_inputs):
            self.node_layers[i] = 0
        
        # All other nodes start at a high layer and are pushed back
        for node_id in self.genome.nodes:
            if node_id not in self.node_layers:
                self.node_layers[node_id] = 1_000_000

        # Iteratively determine layers; this loop is guaranteed to terminate
        for _ in range(len(self.genome.nodes) + 1):
            changed = False
            for gene in self.genome.genes:
                if not gene.enabled: continue
                
                in_layer = self.node_layers.get(gene.in_node, 0)
                out_layer = self.node_layers.get(gene.out_node, 1_000_000)

                new_out_layer = in_layer + 1
                if out_layer > new_out_layer:
                    self.node_layers[gene.out_node] = new_out_layer
                    changed = True
            if not changed: break

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """Activates the network to produce an output for a given sensory input."""
        if len(inputs) != self.genome.num_sensory_inputs:
            raise ValueError(f"Expected {self.genome.num_sensory_inputs} sensory inputs, got {len(inputs)}")
        
        self.nodes = {node_id: 0.0 for node_id in self.genome.nodes}
        
        # Set sensory and bias node values
        for i, value in enumerate(inputs):
            self.nodes[i] = value
        self.nodes[self.genome.bias_node_id] = 1.0
        
        # Propagate through hidden layers
        sorted_layers = sorted(list(set(self.node_layers.values())))
        
        for layer in sorted_layers:
            if layer == 0: continue

            for node_id in [n for n, l in self.node_layers.items() if l == layer]:
                # Skip output nodes in this loop
                if node_id >= self.genome.num_inputs: continue

                node_sum = 0.0
                for gene in self.genome.genes:
                    if gene.enabled and gene.out_node == node_id:
                        node_sum += self.nodes.get(gene.in_node, 0.0) * gene.weight
                
                # Apply activation function only to hidden nodes
                self.nodes[node_id] = math.tanh(node_sum)
        
        outputs = []
        for i in range(self.genome.num_outputs):
            node_id = self.genome.num_inputs + i
            output_sum = 0.0
            for gene in self.genome.genes:
                if gene.enabled and gene.out_node == node_id:
                    output_sum += self.nodes.get(gene.in_node, 0.0) * gene.weight

            outputs.append(output_sum)
        
        return np.array(outputs)

    def get_complexity(self) -> Dict:
        enabled_connections = sum(1 for g in self.genome.genes if g.enabled)
        return {
            'nodes': len(self.genome.nodes),
            'connections': enabled_connections,
            'layers': max(self.node_layers.values()) if self.node_layers else 0
        }