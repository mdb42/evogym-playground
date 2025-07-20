# src/simulation/reporting.py
import pandas as pd
from pathlib import Path
from datetime import datetime

class Reporter:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.generation_data = []
        self.species_data = []
        
        # Define file paths
        self.gen_stats_path = self.output_dir / "generation_stats.csv"
        self.species_stats_path = self.output_dir / "species_stats.csv"
        self.markdown_report_path = self.output_dir / "run_report.md"

        # State for tracking species events
        self.previous_species_ids = set()

    def log_generation(self, generation, population, species_manager):
        """Logs data for a single generation and saves all reports."""
        if not population: return

        # On first generation, clear old report file
        if generation == 0:
            if self.markdown_report_path.exists():
                self.markdown_report_path.unlink()

        # Log to internal data lists for CSVs
        self._log_csv_data(generation, population, species_manager)
        
        # Markdown report
        self._write_markdown_report(generation, population, species_manager)

    def _log_csv_data(self, generation, population, species_manager):
        """Helper to log data for CSV reporting."""
        best_individual = max(population, key=lambda ind: ind.fitness)
        fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
        champion_complexity = best_individual.complexity

        self.generation_data.append({
            'generation': generation, 'best_fitness': best_individual.fitness,
            'avg_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            'num_species': len(species_manager.species) if species_manager else 0,
            'champion_nodes': champion_complexity.get('nodes', 0),
            'champion_conns': champion_complexity.get('connections', 0)
        })

        if species_manager and species_manager.species:
            for sid, species in species_manager.species.items():
                species_fitnesses = [m.fitness for m in species.members if m.fitness is not None]
                best_fitness_in_species = max(species_fitnesses) if species_fitnesses else 0
                self.species_data.append({
                    'generation': generation, 'species_id': f"G{generation - species.age}-{species.id}",
                    'age': species.age, 'stagnation': species.last_improvement,
                    'num_members': len(species.members),
                    'best_fitness_in_species': best_fitness_in_species,
                    'adjusted_fitness': species.get_adjusted_fitness()
                })

        # Update the CSV files
        pd.DataFrame(self.generation_data).to_csv(self.gen_stats_path, index=False)
        pd.DataFrame(self.species_data).to_csv(self.species_stats_path, index=False)

    def _write_markdown_report(self, generation, population, species_manager):
        """Generates and appends a human-readable Markdown report for the generation."""

        report_lines = []
        best_individual = max(population, key=lambda ind: ind.fitness)
        fitnesses = [ind.fitness for ind in population if ind.fitness is not None]

        if generation == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            report_lines.append(f"# NEAT Co-evolution Run: {timestamp}")

        report_lines.append(f"\n\n## Generation {generation}\n")
        report_lines.append(f"- **Best Fitness:** {best_individual.fitness:.4f}")
        report_lines.append(f"- **Average Fitness:** {sum(fitnesses) / len(fitnesses) if fitnesses else 0:.4f}")
        
        if not species_manager:
            report_lines.append("- **Mode:** Random Evolution")
        else:
            report_lines.append(f"- **Number of Species:** {len(species_manager.species)}")
            
            # Track species events
            current_species_ids = set(species_manager.species.keys())
            new_species_ids = current_species_ids - self.previous_species_ids
            extinct_species_ids = self.previous_species_ids - current_species_ids
            
            if new_species_ids or extinct_species_ids:
                report_lines.append("- **Events:**")
                # Find the genome that founded the new species
                for sid in new_species_ids:
                    founder_key = species_manager.species[sid].representative.key
                    report_lines.append(f"  - **!** New Species `{sid}` founded by genome `{founder_key}`.")
                if extinct_species_ids:
                    report_lines.append(f"  - **X** Extinct Species: `{', '.join([str(sid) for sid in extinct_species_ids])}`")

            self.previous_species_ids = current_species_ids
            
            # Species reporting
            if species_manager.species:
                report_lines.append("\n### Species Breakdown")
                
                sorted_species = sorted(
                    species_manager.species.values(),
                    key=lambda s: max([m.fitness for m in s.members if m.fitness is not None], default=-float('inf')),
                    reverse=True
                )

                for species in sorted_species:
                    species_members = sorted(species.members, key=lambda g: g.fitness, reverse=True)
                    best_genome = species_members[0]
                    best_ind_in_species = next((ind for ind in population if ind.genome.key == best_genome.key), None)
                    if best_ind_in_species is None: continue
                    complexity = best_ind_in_species.complexity

                    report_lines.append(f"\n---\n")
                    report_lines.append(f"**Species G{generation - species.age}-{species.id}** (Age: {species.age}, Stagnation: {species.last_improvement}, Members: {len(species.members)})")
                    report_lines.append(f"- **Best Fitness:** {best_genome.fitness:.4f}")
                    report_lines.append(f"- **Adjusted Fitness:** {species.get_adjusted_fitness():.4f}")
                    report_lines.append(f"- **Champion Complexity:** Nodes={complexity['nodes']}, Connections={complexity['connections']}")
                    top_scores = [f"{m.fitness:.4f}" for m in species_members[:5]]
                    report_lines.append(f"- **Fitness Scores:** `[{', '.join(top_scores)}]`")
        
        with open(self.markdown_report_path, "a", encoding="utf-8") as f:
            f.write("\n".join(report_lines))