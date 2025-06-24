import os
import sys
import json
import time
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model, clean_response
from tasks.query import prompt as crossword_query
from prompts.prompt import System_prompt


@dataclass
class PromptCandidate:
    """Represents a candidate system prompt with its performance metrics"""
    prompt: str
    score: float
    generation: int
    id: str
    parent_id: str = None


class CrosswordToTSolver:
    """Tree of Thoughts implementation for evolving system prompts to solve crosswords"""
    
    def __init__(self, target_output_file: str = None):
        self.model = get_model()
        self.target_output_file = target_output_file or "evaluation/results.txt"
        self.target_solution = self._load_target_solution()
        self.generation_log = []
        self.best_prompts_history = []
        
    def _load_target_solution(self) -> List[List[str]]:
        """Load the target crossword solution from results.txt"""
        try:
            with open(self.target_output_file, 'r') as f:
                content = f.read().strip()
            
            # Extract the crossword grid
            lines = content.split('\n')
            grid = []
            for line in lines:
                if line.strip() and not line.startswith("Output:"):
                    # Split by spaces and take only alphabetic characters
                    row = [char.upper() for char in line.split() if char.isalpha()]
                    if len(row) == 5:  # Valid crossword row
                        grid.append(row)
            
            if len(grid) != 5:
                print(f"Warning: Expected 5x5 grid, got {len(grid)} rows")
            
            return grid
        except FileNotFoundError:
            print(f"Target file {self.target_output_file} not found")
            return []
    
    def _generate_initial_prompts(self, n_prompts: int = 3) -> List[PromptCandidate]:
        """Generate initial system prompt variations"""
        base_prompt = System_prompt.strip()
        
        variation_instructions = [
            "Create a variation that emphasizes logical reasoning and step-by-step analysis",
            "Create a variation that focuses on word association and semantic relationships", 
            "Create a variation that emphasizes pattern recognition and constraint satisfaction"
        ]
        
        candidates = []
        
        for i, instruction in enumerate(variation_instructions[:n_prompts]):
            prompt_generation_query = f"""
            Given this base system prompt: "{base_prompt}"
            
            {instruction}
            
            Generate a new system prompt that would be better at solving crossword puzzles.
            The prompt should be clear, specific, and focused on crossword-solving strategies.
            
            Return only the new system prompt, no additional text.
            """
            
            try:
                response = self.model.invoke(prompt_generation_query)
                response = clean_response(response)  # Clean reasoning tags
                new_prompt = response.strip()
                
                candidate = PromptCandidate(
                    prompt=new_prompt,
                    score=0.0,
                    generation=0,
                    id=f"gen0_prompt_{i}",
                    parent_id=None
                )
                candidates.append(candidate)
                
            except Exception as e:
                print(f"Error generating initial prompt {i}: {e}")
                # Fallback to base prompt with modification
                fallback_prompt = f"{base_prompt} Focus on {instruction.lower()}."
                candidate = PromptCandidate(
                    prompt=fallback_prompt,
                    score=0.0,
                    generation=0,
                    id=f"gen0_prompt_{i}_fallback",
                    parent_id=None
                )
                candidates.append(candidate)
        
        return candidates
    
    def _evaluate_prompt(self, candidate: PromptCandidate) -> float:
        """Evaluate a system prompt by testing it on the crossword puzzle"""
        try:
            # Combine system prompt with crossword query
            full_prompt = f"{candidate.prompt}\n\n{crossword_query}"
            
            # Get model response
            response = self.model.invoke(full_prompt)
            response = clean_response(response)  # Clean reasoning tags
            
            # Parse the response to extract the crossword grid
            predicted_grid = self._parse_crossword_response(response)
            
            # Calculate accuracy score
            score = self._calculate_accuracy(predicted_grid, self.target_solution)
            
            # Log the evaluation
            self._log_evaluation(candidate, response, score)
            
            return score
            
        except Exception as e:
            print(f"Error evaluating prompt {candidate.id}: {e}")
            return 0.0
    
    def _parse_crossword_response(self, response: str) -> List[List[str]]:
        """Parse the model response to extract crossword grid"""
        lines = response.strip().split('\n')
        grid = []
        
        for line in lines:
            # Look for lines with 5 space-separated letters
            parts = line.strip().split()
            if len(parts) == 5 and all(len(part) == 1 and part.isalpha() for part in parts):
                grid.append([part.upper() for part in parts])
        
        # If we don't have 5 rows, try to extract from different format
        if len(grid) != 5:
            grid = []
            for line in lines:
                # Try to extract 5 consecutive letters
                letters = ''.join(char.upper() for char in line if char.isalpha())
                if len(letters) >= 5:
                    grid.append(list(letters[:5]))
                if len(grid) == 5:
                    break
        
        # Pad with empty if needed
        while len(grid) < 5:
            grid.append([''] * 5)
        
        return grid
    
    def _calculate_accuracy(self, predicted: List[List[str]], target: List[List[str]]) -> float:
        """Calculate accuracy score between predicted and target grids"""
        if not target or not predicted:
            return 0.0
        
        total_cells = 0
        correct_cells = 0
        
        for i in range(min(len(predicted), len(target))):
            for j in range(min(len(predicted[i]), len(target[i]))):
                total_cells += 1
                if predicted[i][j] == target[i][j]:
                    correct_cells += 1
        
        return correct_cells / total_cells if total_cells > 0 else 0.0
    
    def _log_evaluation(self, candidate: PromptCandidate, response: str, score: float):
        """Log the evaluation details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "candidate_id": candidate.id,
            "generation": candidate.generation,
            "score": score,
            "prompt": candidate.prompt,
            "response": response,
            "target_match": score == 1.0
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Append to log file
        with open("logs/evaluation_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _evolve_prompts(self, parent_candidates: List[PromptCandidate], n_children: int = 2) -> List[PromptCandidate]:
        """Evolve new prompts from parent candidates"""
        new_candidates = []
        
        for parent in parent_candidates:
            for i in range(n_children):
                evolution_query = f"""
                Given this system prompt that achieved a score of {parent.score:.2f} in solving crossword puzzles:
                
                "{parent.prompt}"
                
                Create an improved version that would be better at solving crossword puzzles.
                Consider:
                - More specific crossword-solving strategies
                - Better reasoning approaches
                - Clearer instructions for handling clues
                
                Return only the improved system prompt, no additional text.
                """
                
                try:
                    response = self.model.invoke(evolution_query)
                    response = clean_response(response)  # Clean reasoning tags
                    new_prompt = response.strip()
                    
                    candidate = PromptCandidate(
                        prompt=new_prompt,
                        score=0.0,
                        generation=parent.generation + 1,
                        id=f"gen{parent.generation + 1}_from_{parent.id}_{i}",
                        parent_id=parent.id
                    )
                    new_candidates.append(candidate)
                    
                except Exception as e:
                    print(f"Error evolving prompt from {parent.id}: {e}")
        
        return new_candidates
    
    def _select_top_candidates(self, candidates: List[PromptCandidate], n_select: int) -> List[PromptCandidate]:
        """Select top N candidates based on their scores"""
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:n_select]
    
    def solve_with_tot(self, max_generations: int = 5, initial_candidates: int = 3, 
                      selection_size: int = 2) -> PromptCandidate:
        """
        Main Tree of Thoughts algorithm for prompt evolution
        
        Args:
            max_generations: Maximum number of evolution rounds
            initial_candidates: Number of initial prompt variations to generate
            selection_size: Number of top candidates to select each round
        
        Returns:
            Best performing prompt candidate
        """
        print("🌳 Starting Tree of Thoughts prompt evolution...")
        print(f"Target solution loaded: {len(self.target_solution)} rows")
        
        # Generate initial candidates
        print(f"\n🌱 Generating {initial_candidates} initial prompt candidates...")
        current_candidates = self._generate_initial_prompts(initial_candidates)
        
        # Evaluate initial candidates
        print("📊 Evaluating initial candidates...")
        for candidate in current_candidates:
            candidate.score = self._evaluate_prompt(candidate)
            print(f"  {candidate.id}: {candidate.score:.3f}")
        
        # Evolution loop
        for generation in range(max_generations):
            print(f"\n🔄 Generation {generation + 1}")
            
            # Select top candidates
            selected = self._select_top_candidates(current_candidates, selection_size)
            print(f"Selected top {len(selected)} candidates:")
            for candidate in selected:
                print(f"  {candidate.id}: {candidate.score:.3f}")
            
            # Store best candidates for this generation
            self.best_prompts_history.append({
                "generation": generation,
                "candidates": [(c.id, c.score, c.prompt) for c in selected]
            })
            
            # Check if we found perfect solution
            if selected[0].score == 1.0:
                print(f"🎯 Perfect solution found in generation {generation}!")
                return selected[0]
            
            # Evolve new candidates
            print("🧬 Evolving new candidates...")
            new_candidates = self._evolve_prompts(selected)
            
            # Evaluate new candidates
            print("📊 Evaluating new candidates...")
            for candidate in new_candidates:
                candidate.score = self._evaluate_prompt(candidate)
                print(f"  {candidate.id}: {candidate.score:.3f}")
            
            # Combine with previous generation for next iteration
            current_candidates = selected + new_candidates
        
        # Return best candidate
        best_candidate = self._select_top_candidates(current_candidates, 1)[0]
        print(f"\n🏆 Best candidate after {max_generations} generations:")
        print(f"  ID: {best_candidate.id}")
        print(f"  Score: {best_candidate.score:.3f}")
        print(f"  Prompt: {best_candidate.prompt}")
        
        return best_candidate
    
    def save_results(self, best_candidate: PromptCandidate):
        """Save the results to files"""
        results = {
            "best_prompt": {
                "id": best_candidate.id,
                "score": best_candidate.score,
                "generation": best_candidate.generation,
                "prompt": best_candidate.prompt,
                "parent_id": best_candidate.parent_id
            },
            "evolution_history": self.best_prompts_history,
            "target_solution": self.target_solution,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to JSON file
        with open("logs/tot_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Update the original prompt file with the best prompt
        with open("prompts/prompt.py", "w") as f:
            f.write(f'System_prompt = """{best_candidate.prompt}"""\n\n')
        
        print(f"Results saved to logs/tot_results.json")
        print(f"Best prompt saved to prompts/prompt.py")


def main():
    """Main function to run the Tree of Thoughts prompt evolution"""
    print("🎯 Crossword Solver - Tree of Thoughts Prompt Evolution")
    print("=" * 60)
    
    # Initialize solver
    solver = CrosswordToTSolver()
    
    # Run Tree of Thoughts evolution
    best_prompt = solver.solve_with_tot(
        max_generations=3,
        initial_candidates=3,
        selection_size=2
    )
    
    # Save results
    solver.save_results(best_prompt)
    
    print("\n✅ Tree of Thoughts evolution completed!")
    print(f"Best prompt achieved {best_prompt.score:.1%} accuracy")


if __name__ == "__main__":
    main()
