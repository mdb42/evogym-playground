# src/neat/network.py
import numpy as np
from typing import Dict, List
from .neat import NEATGenome, Gene

class NEATNetwork:    
    def __init__(self, genome: NEATGenome):
        self.genome = genome
        self.nodes = {}  # node_id -> activation_value
        self.node_layers = {}  # node_id -> layer_number
        
        # Build network structure
        self._build_layers()

    def _build_layers(self):
        # Determine which layer each node belongs to
        # Input nodes are layer 0
        for i in range(self.genome.num_inputs):
            self.node_layers[i] = 0
        
        # Output nodes start at layer 1
        for i in range(self.genome.num_outputs):
            node_id = self.genome.num_inputs + i
            self.node_layers[node_id] = 1
        
        for _ in range(len(self.genome.nodes) + 1):
            for gene in self.genome.genes:
                if not gene.enabled:
                    continue
                
                if gene.in_node in self.node_layers:
                    # Default the layer for the output node if it doesn't exist yet
                    if gene.out_node not in self.node_layers:
                        self.node_layers[gene.out_node] = 1

                    # Push the output node to a later layer if necessary
                    new_out_layer = self.node_layers[gene.in_node] + 1
                    if self.node_layers[gene.out_node] < new_out_layer:
                        self.node_layers[gene.out_node] = new_out_layer
    
    def activate(self, inputs: np.ndarray) -> np.ndarray:
        if len(inputs) != self.genome.num_inputs:
            raise ValueError(f"Expected {self.genome.num_inputs} inputs, got {len(inputs)}")
        
        # Reset all node values
        self.nodes = {node: 0.0 for node in self.genome.nodes}
        
        # Set input values
        for i, value in enumerate(inputs):
            self.nodes[i] = value
        
        # Check if we have any enabled connections
        enabled_connections = [g for g in self.genome.genes if g.enabled]
        if not enabled_connections:
            return np.zeros(self.genome.num_outputs)
        
        # Get maximum layer
        max_layer = max(self.node_layers.values()) if self.node_layers else 0
        
        # Process layer by layer
        for layer in range(1, max_layer + 1):
            layer_nodes = [n for n, l in self.node_layers.items() if l == layer]
            
            for node in layer_nodes:
                node_sum = 0.0
                for gene in self.genome.genes:
                    if gene.enabled and gene.out_node == node:
                        if gene.in_node in self.nodes:
                            node_sum += self.nodes[gene.in_node] * gene.weight
                
                # Apply activation function
                self.nodes[node] = np.tanh(node_sum)
        
        # Extract outputs
        outputs = []
        for i in range(self.genome.num_outputs):
            node_id = self.genome.num_inputs + i
            outputs.append(self.nodes.get(node_id, 0.0))
        
        return np.array(outputs)

    def get_complexity(self) -> Dict:
        enabled_connections = sum(1 for g in self.genome.genes if g.enabled)
        return {
            'nodes': len(self.genome.nodes),
            'connections': enabled_connections,
            'layers': max(self.node_layers.values()) if self.node_layers else 0
        }