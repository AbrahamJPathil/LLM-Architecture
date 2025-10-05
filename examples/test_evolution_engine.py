"""
Unit Tests for PromptEvolve - Evolution Engine and Prompt Verifier

This module contains comprehensive unit tests for the core components of the
PromptEvolve system, focusing on the PromptVerifier (scoring/evaluation logic)
and EvolutionEngine (selection and combination logic).
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import statistics
from typing import List

# Import components to test
import sys
import os
sys.path.append(os.path.dirname(__file__))

from prompt_evolution import (
    PromptResult,
    TestScenario,
    EvolutionHistory,
    PromptState,
    DomainConfig,
    PromptVerifier,
    EvolutionEngine,
)


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

class TestFixtures:
    """Common test data fixtures."""
    
    @staticmethod
    def create_test_scenarios() -> List[TestScenario]:
        """Create sample test scenarios for testing."""
        return [
            TestScenario(
                input_message="What is 2+2?",
                existing_memories="",
                desired_output="2+2 equals 4.",
                bad_output="2+2 equals 5.",
                metadata={"difficulty": "easy"}
            ),
            TestScenario(
                input_message="Explain quantum physics",
                existing_memories="User has PhD in physics",
                desired_output="Quantum physics describes behavior at atomic scale...",
                bad_output="Quantum physics is magic.",
                metadata={"difficulty": "hard"}
            ),
            TestScenario(
                input_message="What's the capital of France?",
                existing_memories="",
                desired_output="The capital of France is Paris.",
                bad_output="The capital of France is London.",
                metadata={"difficulty": "easy"}
            )
        ]
    
    @staticmethod
    def create_prompt_results() -> List[PromptResult]:
        """Create sample prompt results for testing."""
        return [
            PromptResult(
                prompt="You are a helpful assistant.",
                response="Sample response 1",
                success_rate=0.9,
                quality_score=0.85,
                consistency_score=0.88,
                efficiency_score=0.92,
                execution_time=1.5,
                domain_metrics={"accuracy": 0.9},
                test_count=10
            ),
            PromptResult(
                prompt="You are a precise assistant.",
                response="Sample response 2",
                success_rate=0.75,
                quality_score=0.70,
                consistency_score=0.72,
                efficiency_score=0.80,
                execution_time=2.0,
                domain_metrics={"accuracy": 0.75},
                test_count=10
            ),
            PromptResult(
                prompt="You are an expert assistant.",
                response="Sample response 3",
                success_rate=0.95,
                quality_score=0.92,
                consistency_score=0.90,
                efficiency_score=0.88,
                execution_time=1.8,
                domain_metrics={"accuracy": 0.95},
                test_count=10
            ),
            PromptResult(
                prompt="You are a basic assistant.",
                response="Sample response 4",
                success_rate=0.60,
                quality_score=0.55,
                consistency_score=0.65,
                efficiency_score=0.70,
                execution_time=2.5,
                domain_metrics={"accuracy": 0.60},
                test_count=10
            ),
            PromptResult(
                prompt="You are a detailed assistant.",
                response="Sample response 5",
                success_rate=0.85,
                quality_score=0.80,
                consistency_score=0.82,
                efficiency_score=0.85,
                execution_time=1.7,
                domain_metrics={"accuracy": 0.85},
                test_count=10
            ),
        ]
    
    @staticmethod
    def create_domain_config() -> DomainConfig:
        """Create a sample domain configuration."""
        return DomainConfig(
            domain_name="test_domain",
            base_prompt_template="You are a test assistant.",
            evaluation_criteria={
                "accuracy": 0.4,
                "completeness": 0.3,
                "clarity": 0.2,
                "efficiency": 0.1
            },
            thresholds={
                "success_rate": 0.80,
                "quality_score": 0.75
            },
            test_scenario_count=10
        )


# ============================================================================
# PROMPT RESULT TESTS
# ============================================================================

class TestPromptResult(unittest.TestCase):
    """Test cases for PromptResult model and methods."""
    
    def test_prompt_result_creation(self):
        """Test creating a valid PromptResult."""
        result = PromptResult(
            prompt="Test prompt",
            response="Test response",
            success_rate=0.85,
            quality_score=0.80,
            execution_time=1.5,
            test_count=10
        )
        
        self.assertEqual(result.success_rate, 0.85)
        self.assertEqual(result.quality_score, 0.80)
        self.assertEqual(result.test_count, 10)
    
    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        result = PromptResult(
            prompt="Test prompt",
            response="Test response",
            success_rate=0.8,  # 40%
            quality_score=0.7,  # 30%
            consistency_score=0.9,  # 20%
            efficiency_score=0.6,  # 10%
            execution_time=1.5,
            test_count=10
        )
        
        expected = (0.8 * 0.4) + (0.7 * 0.3) + (0.9 * 0.2) + (0.6 * 0.1)
        self.assertAlmostEqual(result.get_composite_score(), expected, places=4)
    
    def test_invalid_success_rate(self):
        """Test that invalid success rates are rejected."""
        with self.assertRaises(Exception):  # Pydantic validation error
            PromptResult(
                prompt="Test prompt",
                response="Test response",
                success_rate=1.5,  # Invalid: > 1.0
                quality_score=0.80,
                execution_time=1.5,
                test_count=10
            )
    
    def test_prompt_too_short(self):
        """Test that prompts that are too short are rejected."""
        with self.assertRaises(Exception):  # Pydantic validation error
            PromptResult(
                prompt="Hi",  # Too short
                response="Test response",
                success_rate=0.85,
                quality_score=0.80,
                execution_time=1.5,
                test_count=10
            )


# ============================================================================
# EVOLUTION ENGINE TESTS
# ============================================================================

class TestEvolutionEngine(unittest.TestCase):
    """Test cases for EvolutionEngine selection and iteration logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'evolution': {
                'max_iterations': 10,
                'min_iterations': 3,
                'selection_top_percentage': 0.30,
                'min_selection_count': 2,
                'enable_crossover': True,
                'population_size': 5
            },
            'thresholds': {
                'success_rate': 0.85,
                'quality_score': 0.80,
                'min_improvement_threshold': 0.02
            }
        }
        self.logger = Mock()
        self.engine = EvolutionEngine(self.config, self.logger)
        self.fixtures = TestFixtures()
    
    def test_select_top_performers_default_percentage(self):
        """Test selecting top performers with default selection percentage."""
        results = self.fixtures.create_prompt_results()
        
        selected = self.engine.select_top_performers(results)
        
        # Should select top 30% (1.5 -> 2 results due to min_selection_count)
        self.assertEqual(len(selected), 2)
        
        # Should be sorted by composite score (highest first)
        self.assertTrue(selected[0].get_composite_score() >= selected[1].get_composite_score())
    
    def test_select_top_performers_custom_percentage(self):
        """Test selecting top performers with custom selection percentage."""
        results = self.fixtures.create_prompt_results()
        
        selected = self.engine.select_top_performers(results, selection_percentage=0.5)
        
        # Should select top 50% (2.5 -> 3 results)
        self.assertGreaterEqual(len(selected), 2)
    
    def test_select_top_performers_respects_minimum(self):
        """Test that selection respects minimum selection count."""
        results = self.fixtures.create_prompt_results()[:2]  # Only 2 results
        
        selected = self.engine.select_top_performers(results, selection_percentage=0.1)
        
        # Should select min_selection_count (2) even though 10% would be 0.2
        self.assertEqual(len(selected), 2)
    
    def test_select_top_performers_empty_list_raises_error(self):
        """Test that empty results list raises an error."""
        with self.assertRaises(ValueError):
            self.engine.select_top_performers([])
    
    def test_select_top_performers_ordering(self):
        """Test that selected prompts are ordered by performance."""
        results = self.fixtures.create_prompt_results()
        
        selected = self.engine.select_top_performers(results)
        
        # Verify ordering by composite score
        scores = [r.get_composite_score() for r in selected]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_combine_prompts(self):
        """Test combining two prompts."""
        prompt1 = "You are helpful.\nBe concise."
        prompt2 = "You are helpful.\nBe detailed."
        learnings = ["Focus on clarity"]
        
        combined = self.engine.combine_prompts(prompt1, prompt2, learnings)
        
        # Should contain elements from both prompts
        self.assertIsInstance(combined, str)
        self.assertGreater(len(combined), 0)
    
    def test_should_continue_max_iterations(self):
        """Test that loop stops at max iterations."""
        result = self.fixtures.create_prompt_results()[0]
        history = []
        
        should_continue, reason = self.engine.should_continue(10, result, history)
        
        self.assertFalse(should_continue)
        self.assertIn("maximum iterations", reason.lower())
    
    def test_should_continue_before_min_iterations(self):
        """Test that loop continues before min iterations."""
        result = self.fixtures.create_prompt_results()[0]
        history = []
        
        should_continue, reason = self.engine.should_continue(2, result, history)
        
        self.assertTrue(should_continue)
        self.assertIsNone(reason)
    
    def test_should_continue_thresholds_met(self):
        """Test that loop stops when thresholds are met."""
        result = PromptResult(
            prompt="Excellent prompt",
            response="Test",
            success_rate=0.90,  # Above threshold (0.85)
            quality_score=0.85,  # Above threshold (0.80)
            execution_time=1.0,
            test_count=10
        )
        history = [
            EvolutionHistory(
                iteration=1,
                prompts_tested=5,
                max_success_rate=0.7,
                avg_success_rate=0.6,
                max_quality_score=0.65,
                avg_quality_score=0.6,
                best_prompt_preview="Preview",
                learnings=[]
            ),
            EvolutionHistory(
                iteration=2,
                prompts_tested=5,
                max_success_rate=0.8,
                avg_success_rate=0.75,
                max_quality_score=0.75,
                avg_quality_score=0.7,
                best_prompt_preview="Preview",
                learnings=[]
            ),
            EvolutionHistory(
                iteration=3,
                prompts_tested=5,
                max_success_rate=0.85,
                avg_success_rate=0.8,
                max_quality_score=0.80,
                avg_quality_score=0.75,
                best_prompt_preview="Preview",
                learnings=[]
            )
        ]
        
        should_continue, reason = self.engine.should_continue(4, result, history)
        
        self.assertFalse(should_continue)
        self.assertIn("threshold", reason.lower())
    
    def test_should_continue_plateau_detection(self):
        """Test that loop stops when improvement plateaus."""
        result = PromptResult(
            prompt="Test prompt",
            response="Test",
            success_rate=0.75,
            quality_score=0.70,
            execution_time=1.0,
            test_count=10
        )
        
        # Create history showing plateau (minimal improvements)
        history = [
            EvolutionHistory(
                iteration=i,
                prompts_tested=5,
                max_success_rate=0.70,
                avg_success_rate=0.65,
                max_quality_score=0.70 + (i * 0.005),  # Very small improvements
                avg_quality_score=0.65,
                best_prompt_preview="Preview",
                learnings=[]
            )
            for i in range(1, 6)
        ]
        
        should_continue, reason = self.engine.should_continue(6, result, history)
        
        self.assertFalse(should_continue)
        self.assertIn("plateau", reason.lower())


# ============================================================================
# PROMPT VERIFIER TESTS (MOCKED)
# ============================================================================

class TestPromptVerifier(unittest.TestCase):
    """Test cases for PromptVerifier evaluation logic."""
    
    def setUp(self):
        """Set up test fixtures with mocked OpenAI client."""
        self.config = {
            'models': {
                'prompt_verifier': {
                    'name': 'gpt-4-turbo-preview',
                    'temperature': 0.2,
                    'max_tokens': 1000
                }
            }
        }
        self.logger = Mock()
        self.mock_client = Mock()
        self.verifier = PromptVerifier(self.mock_client, self.config, self.logger)
        self.fixtures = TestFixtures()
    
    def test_evaluate_results_with_perfect_scores(self):
        """Test evaluation with perfect judgment responses."""
        # Mock the LLM judgment responses
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "is_success": true,
            "quality_score": 1.0,
            "feedback": "Perfect response"
        }
        '''
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Create test data
        prompt_result = PromptResult(
            prompt="Test prompt",
            response="Test response",
            success_rate=0.0,  # Will be updated by verifier
            quality_score=0.0,  # Will be updated by verifier
            execution_time=1.5,
            test_count=3,
            domain_metrics={
                'raw_results': [
                    {'response': 'Good', 'execution_time': 1.5, 'desired_output': 'Good', 'bad_output': 'Bad'},
                    {'response': 'Good', 'execution_time': 1.5, 'desired_output': 'Good', 'bad_output': 'Bad'},
                    {'response': 'Good', 'execution_time': 1.5, 'desired_output': 'Good', 'bad_output': 'Bad'}
                ]
            }
        )
        
        test_scenarios = self.fixtures.create_test_scenarios()
        
        # Evaluate
        updated_result, learnings = self.verifier.evaluate_results(
            prompt_result,
            test_scenarios
        )
        
        # Verify results
        self.assertEqual(updated_result.success_rate, 1.0)  # All succeeded
        self.assertEqual(updated_result.quality_score, 1.0)  # Perfect quality
        self.assertGreater(len(learnings), 0)  # Should have learnings
    
    def test_evaluate_results_with_mixed_scores(self):
        """Test evaluation with mixed success/failure."""
        # Mock alternating success/failure responses
        def side_effect(*args, **kwargs):
            response = Mock()
            response.choices = [Mock()]
            # Alternate between success and failure
            if side_effect.call_count % 2 == 1:
                response.choices[0].message.content = '{"is_success": true, "quality_score": 0.8, "feedback": "Good"}'
            else:
                response.choices[0].message.content = '{"is_success": false, "quality_score": 0.4, "feedback": "Needs improvement"}'
            side_effect.call_count += 1
            return response
        
        side_effect.call_count = 0
        self.mock_client.chat.completions.create.side_effect = side_effect
        
        prompt_result = PromptResult(
            prompt="Test prompt",
            response="Test response",
            success_rate=0.0,
            quality_score=0.0,
            execution_time=1.5,
            test_count=4,
            domain_metrics={
                'raw_results': [
                    {'response': f'Response {i}', 'execution_time': 1.5, 'desired_output': 'Good', 'bad_output': 'Bad'}
                    for i in range(4)
                ]
            }
        )
        
        test_scenarios = self.fixtures.create_test_scenarios()[:4]
        
        updated_result, learnings = self.verifier.evaluate_results(
            prompt_result,
            test_scenarios
        )
        
        # Should have 50% success rate (2 out of 4)
        self.assertEqual(updated_result.success_rate, 0.5)
        self.assertGreater(updated_result.quality_score, 0.0)
        self.assertLess(updated_result.quality_score, 1.0)
    
    def test_evaluate_results_empty_results(self):
        """Test evaluation with no results."""
        prompt_result = PromptResult(
            prompt="Test prompt",
            response="",
            success_rate=0.0,
            quality_score=0.0,
            execution_time=0.0,
            test_count=0,
            domain_metrics={'raw_results': []}
        )
        
        test_scenarios = []
        
        updated_result, learnings = self.verifier.evaluate_results(
            prompt_result,
            test_scenarios
        )
        
        self.assertIn("No results", learnings[0])
    
    def test_evaluate_results_with_domain_config(self):
        """Test evaluation with domain-specific configuration."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"is_success": true, "quality_score": 0.9, "feedback": "Excellent"}'
        self.mock_client.chat.completions.create.return_value = mock_response
        
        prompt_result = PromptResult(
            prompt="Test prompt",
            response="Test response",
            success_rate=0.0,
            quality_score=0.0,
            execution_time=1.5,
            test_count=2,
            domain_metrics={
                'raw_results': [
                    {'response': 'Good', 'execution_time': 1.5, 'desired_output': 'Good', 'bad_output': 'Bad'},
                    {'response': 'Good', 'execution_time': 1.5, 'desired_output': 'Good', 'bad_output': 'Bad'}
                ]
            }
        )
        
        test_scenarios = self.fixtures.create_test_scenarios()[:2]
        domain_config = self.fixtures.create_domain_config()
        
        updated_result, learnings = self.verifier.evaluate_results(
            prompt_result,
            test_scenarios,
            domain_config
        )
        
        # Verify that domain config was used (check mock calls)
        self.assertTrue(self.mock_client.chat.completions.create.called)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEvolutionIntegration(unittest.TestCase):
    """Integration tests for the evolution process."""
    
    def test_full_selection_pipeline(self):
        """Test the complete selection pipeline."""
        config = {
            'evolution': {
                'max_iterations': 10,
                'min_iterations': 3,
                'selection_top_percentage': 0.40,
                'min_selection_count': 2,
                'enable_crossover': True,
                'population_size': 5
            },
            'thresholds': {
                'success_rate': 0.85,
                'quality_score': 0.80,
                'min_improvement_threshold': 0.02
            }
        }
        logger = Mock()
        engine = EvolutionEngine(config, logger)
        fixtures = TestFixtures()
        
        # Get results
        results = fixtures.create_prompt_results()
        
        # Select top performers
        selected = engine.select_top_performers(results)
        
        # Verify selection
        self.assertGreater(len(selected), 0)
        self.assertLessEqual(len(selected), len(results))
        
        # Verify all selected have valid scores
        for result in selected:
            self.assertGreaterEqual(result.get_composite_score(), 0.0)
            self.assertLessEqual(result.get_composite_score(), 1.0)
        
        # Test combination
        if len(selected) >= 2:
            combined = engine.combine_prompts(
                selected[0].prompt,
                selected[1].prompt,
                ["Test learning"]
            )
            self.assertIsInstance(combined, str)
            self.assertGreater(len(combined), 0)


# ============================================================================
# RUN TESTS
# ============================================================================

def run_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPromptResult))
    suite.addTests(loader.loadTestsFromTestCase(TestEvolutionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestPromptVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestEvolutionIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
