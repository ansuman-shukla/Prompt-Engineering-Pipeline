# Tree of Thoughts Prompt Evolution for Crossword Solving

This implementation uses Tree of Thoughts (ToT) to evolve system prompts that are better at solving crossword puzzles.

## How it works

1. **Initial Generation**: Creates 3 different variations of the base system prompt
2. **Evaluation**: Tests each prompt by having it solve the crossword puzzle and comparing with the target solution
3. **Selection**: Keeps the top 2 performing prompts
4. **Evolution**: Generates 2 new variations from each selected prompt
5. **Repeat**: Continues for 5 generations to find the optimal prompt

## Tree Structure

```
Generation 0: [Prompt A] [Prompt B] [Prompt C]
                  ↓         ↓
Generation 1: [A1] [A2] [B1] [B2]  (top 2 selected: A, B)
                  ↓         ↓
Generation 2: [A1.1] [A1.2] [B1.1] [B1.2]
                  ↓             ↓
Generation 3: [A1.1.1] [A1.1.2] [B1.1.1] [B1.1.2]
                  ↓
Generation 4: [Best evolved prompts...]
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your environment
Make sure you have your model configured in `src/model.py`. This implementation is configured for:
- **qwen3:4b** model running locally via Ollama
- Handles reasoning models that output `<think>` tags
- Automatically cleans responses to extract actual content

### 3. Test your model setup (optional)
```bash
python test_model.py
```

### 4. Prepare your target solution
Place the correct crossword solution in `evaluation/results.txt` in this format:
```
Output:

R I L L E
O L E I N  
T E M P T
A B A S E
L O N E R
```

### 5. Run the evolution
```bash
cd src
python main.py
```

### 6. Run tests (optional)
```bash
python test_tot.py
```

## Model Compatibility

This implementation is specifically designed to work with reasoning models like **qwen3:4b** that output responses in this format:

```
<think>
Internal reasoning and thought process...
</think>

Actual response content here.
```

The system automatically:
- Removes `<think>...</think>` blocks from responses
- Extracts the actual content for evaluation
- Handles cases where the entire response is wrapped in thinking tags

## Output Files

- `logs/evaluation_log.json`: Detailed log of all evaluations
- `logs/tot_results.json`: Final results and evolution history
- `prompts/prompt.py`: Updated with the best prompt found

## Configuration

You can modify the algorithm parameters in `main.py`:

```python
best_prompt = solver.solve_with_tot(
    max_generations=5,      # Number of evolution rounds
    initial_candidates=3,   # Initial prompt variations
    selection_size=2        # Top N to keep each round
)
```

## Key Classes

- `PromptCandidate`: Represents a prompt with its score and metadata
- `CrosswordToTSolver`: Main class implementing the ToT algorithm

## Evaluation Metrics

The system evaluates prompts based on exact character match with the target crossword solution:
- Score = (Correct letters) / (Total letters)
- Perfect score = 1.0 (100% match)

## Tree of Thoughts Benefits

1. **Exploration**: Tests multiple prompt strategies simultaneously
2. **Evolution**: Improves prompts based on performance feedback  
3. **Selection**: Keeps only the best performing approaches
4. **Convergence**: Iteratively improves to find optimal prompts

This approach is much more effective than manual prompt engineering because it:
- Tests multiple approaches in parallel
- Uses objective performance metrics
- Automatically evolves better solutions
- Avoids local optima through diverse exploration
