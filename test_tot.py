#!/usr/bin/env python3
"""
Test script to verify the Tree of Thoughts implementation works correctly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import CrosswordToTSolver

def test_target_loading():
    """Test if target solution loads correctly"""
    print("Testing target solution loading...")
    solver = CrosswordToTSolver()
    print(f"Target solution: {solver.target_solution}")
    print(f"Grid size: {len(solver.target_solution)}x{len(solver.target_solution[0]) if solver.target_solution else 0}")
    return len(solver.target_solution) == 5

def test_prompt_generation():
    """Test initial prompt generation"""
    print("\nTesting prompt generation...")
    solver = CrosswordToTSolver()
    candidates = solver._generate_initial_prompts(2)
    print(f"Generated {len(candidates)} candidates")
    for i, candidate in enumerate(candidates):
        print(f"Candidate {i+1}: {candidate.prompt[:100]}...")
    return len(candidates) == 2

def test_evaluation():
    """Test prompt evaluation"""
    print("\nTesting prompt evaluation...")
    solver = CrosswordToTSolver()
    candidates = solver._generate_initial_prompts(1)
    if candidates:
        score = solver._evaluate_prompt(candidates[0])
        print(f"Evaluation score: {score}")
        return True
    return False

if __name__ == "__main__":
    print("🧪 Running Tree of Thoughts tests...")
    print("=" * 50)
    
    tests = [
        ("Target Loading", test_target_loading),
        ("Prompt Generation", test_prompt_generation), 
        ("Evaluation", test_evaluation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Ready to run the main algorithm.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
