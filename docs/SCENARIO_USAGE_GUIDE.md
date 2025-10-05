# Test Scenarios Guide

## How Test Scenarios Work in PromptEvolve

### 1. What is a Test Scenario?

A **TestScenario** is a structured test case that defines:
- **Input**: What the user asks
- **Context**: Any background information
- **Expected Output**: What a good response looks like
- **Bad Output**: What to avoid
- **Metadata**: Additional info (difficulty, domain, etc.)

```python
class TestScenario(BaseModel):
    input_message: str           # The user's question/request
    existing_memories: str       # Context or conversation history
    desired_output: str          # The ideal response
    bad_output: str             # A counter-example (what NOT to say)
    metadata: Dict[str, Any]    # Extra info like {"difficulty": "hard"}
```

---

## 2. The France & Quantum Examples (YES, I PROVIDED THEM!)

These are **hardcoded examples** in the `main()` function at the bottom of `prompt_evolution.py`:

```python
def main():
    """Example usage of the PromptEvolution system."""
    
    # Initialize the system
    system = PromptEvolution(config_path="config.yaml")
    
    # Create sample test scenarios
    test_scenarios = [
        TestScenario(
            input_message="What is the capital of France?",
            existing_memories="",
            desired_output="The capital of France is Paris.",
            bad_output="The capital is London.",
            metadata={"difficulty": "easy"}
        ),
        TestScenario(
            input_message="Explain quantum entanglement",
            existing_memories="User has basic physics knowledge",
            desired_output="Quantum entanglement is a phenomenon where particles become correlated...",
            bad_output="Entanglement is magic.",
            metadata={"difficulty": "hard"}
        )
    ]
    
    # Base prompt to optimize
    base_prompt = "You are a helpful assistant. Answer questions accurately and concisely."
    
    # Run evolution
    final_state = system.evolve_prompt(
        base_prompt=base_prompt,
        test_scenarios=test_scenarios
    )
```

**These are just demo examples!** In production, you would replace them with real domain-specific scenarios.

---

## 3. How the System Uses Test Scenarios

### Step-by-Step Execution:

#### **Step 1: Execute Prompt Against Each Scenario**

```python
def execute_prompt(self, prompt, test_scenarios):
    results = []
    
    for scenario in test_scenarios:
        # Construct the message
        user_message = scenario.input_message
        if scenario.existing_memories:
            user_message = f"Context: {scenario.existing_memories}\n\nQuery: {user_message}"
        
        # Call the LLM
        response = call_llm(
            client=self.client,
            provider=self.provider,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        # Store result
        results.append({
            'response': response,
            'desired_output': scenario.desired_output,
            'bad_output': scenario.bad_output
        })
```

**Example Execution:**

For the France scenario:
```
System: "You are a helpful assistant. Answer questions accurately and concisely."
User: "What is the capital of France?"

LLM Response: "The capital of France is Paris."
```

For the Quantum scenario:
```
System: "You are a helpful assistant. Answer questions accurately and concisely."
User: "Context: User has basic physics knowledge\n\nQuery: Explain quantum entanglement"

LLM Response: "Quantum entanglement is a phenomenon where two or more particles..."
```

---

#### **Step 2: Evaluate Each Response**

The system uses **another LLM call** to judge if the response is good:

```python
def _judge_response(self, response, desired, bad_example):
    # Ask the LLM to judge the quality
    judgment_prompt = f"""
    Evaluate this response:
    
    Actual Response: {response}
    Desired Output: {desired}
    Bad Example: {bad_example}
    
    Is the actual response:
    1. Correct and aligned with desired output?
    2. Avoiding the bad example pattern?
    
    Provide a quality score (0.0 to 1.0) and feedback.
    """
    
    # LLM judges the response
    judgment = call_llm(...)
    
    return {
        'is_success': True/False,
        'quality_score': 0.95,
        'feedback': "The response correctly states Paris is the capital..."
    }
```

---

#### **Step 3: Calculate Aggregate Metrics**

```python
success_rate = 2/2 = 100%  # Both scenarios passed
quality_score = (1.0 + 0.9) / 2 = 0.95
consistency_score = 0.964  # Low variance in scores
efficiency_score = 0.854   # Based on response time
```

---

## 4. How to Use Test Scenarios in Real Applications

### **Example 1: Legal Domain**

```python
legal_scenarios = [
    TestScenario(
        input_message="What is the statute of limitations for breach of contract in California?",
        existing_memories="User is a paralegal working on civil litigation",
        desired_output="In California, the statute of limitations for breach of written contract is 4 years (CCP § 337), and for oral contracts is 2 years (CCP § 339).",
        bad_output="The statute of limitations is 5 years.",
        metadata={"domain": "legal", "difficulty": "medium", "jurisdiction": "California"}
    ),
    TestScenario(
        input_message="Can I file a lawsuit after the statute of limitations has expired?",
        existing_memories="",
        desired_output="Generally, no. Once the statute of limitations has expired, the defendant can file a motion to dismiss based on the statute of limitations defense...",
        bad_output="Yes, you can always file a lawsuit.",
        metadata={"domain": "legal", "difficulty": "easy"}
    )
]

base_prompt = "You are a legal assistant specialized in California civil law. Provide accurate legal information with statute citations."

system = PromptEvolution(config_path="config.yaml")
evolved_prompt = system.evolve_prompt(
    base_prompt=base_prompt,
    test_scenarios=legal_scenarios
)
```

---

### **Example 2: Medical Domain**

```python
medical_scenarios = [
    TestScenario(
        input_message="What are the symptoms of type 2 diabetes?",
        existing_memories="Patient is 45 years old, overweight",
        desired_output="Common symptoms of type 2 diabetes include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow-healing sores, and frequent infections. However, many people with type 2 diabetes have no symptoms. I recommend consulting with a healthcare provider for proper diagnosis.",
        bad_output="You definitely have diabetes. Start taking insulin immediately.",
        metadata={"domain": "medical", "difficulty": "medium"}
    )
]
```

---

### **Example 3: Customer Support**

```python
support_scenarios = [
    TestScenario(
        input_message="I want a refund for my order",
        existing_memories="Order #12345, purchased 5 days ago, product: Headphones",
        desired_output="I'd be happy to help you with a refund for order #12345. According to our policy, products can be returned within 30 days of purchase. I'll initiate the refund process. You should see the amount credited to your original payment method within 5-7 business days. Would you like me to also send you a prepaid return shipping label?",
        bad_output="No refunds allowed.",
        metadata={"domain": "support", "sentiment": "negative"}
    )
]
```

---

## 5. How to Load Scenarios from Files

### **Option A: JSON File**

Create `data/test_scenarios/legal/contract_scenarios.json`:

```json
[
  {
    "input_message": "What is breach of contract?",
    "existing_memories": "",
    "desired_output": "A breach of contract occurs when one party fails to perform their obligations under the contract...",
    "bad_output": "It means someone broke a promise.",
    "metadata": {"difficulty": "easy"}
  },
  {
    "input_message": "What remedies are available for breach of contract?",
    "existing_memories": "User is studying contract law",
    "desired_output": "Common remedies include: 1) Damages (compensatory, consequential, liquidated), 2) Specific performance, 3) Rescission, 4) Reformation...",
    "bad_output": "You can sue them.",
    "metadata": {"difficulty": "hard"}
  }
]
```

Then load them:

```python
import json
from pathlib import Path

def load_scenarios_from_file(filepath: str) -> List[TestScenario]:
    """Load test scenarios from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [TestScenario(**scenario) for scenario in data]

# Usage
scenarios = load_scenarios_from_file("data/test_scenarios/legal/contract_scenarios.json")

system = PromptEvolution(config_path="config.yaml")
result = system.evolve_prompt(
    base_prompt="You are a legal assistant...",
    test_scenarios=scenarios
)
```

---

### **Option B: Generate with Task Definer**

```python
from taskdefiner import TaskDefiner

# Define the task
definer = TaskDefiner()
task = definer.define_task(
    task_description="Create a legal assistant for contract law questions",
    domain="legal"
)

# Generate test scenarios from the task
scenarios = generate_scenarios_from_task(task)  # We need to build this function!

# Evolve the prompt
system = PromptEvolution(config_path="config.yaml")
result = system.evolve_prompt(
    base_prompt=task.base_prompt,
    test_scenarios=scenarios
)
```

---

## 6. Full Workflow Example

```python
#!/usr/bin/env python3
"""
Full workflow: Load scenarios, evolve prompt, save results
"""

import json
from pathlib import Path
from prompt_evolution import PromptEvolution, TestScenario

def main():
    # 1. Load test scenarios from file
    scenarios_file = "data/test_scenarios/legal/contract_law.json"
    with open(scenarios_file, 'r') as f:
        scenario_data = json.load(f)
    
    test_scenarios = [TestScenario(**s) for s in scenario_data]
    
    # 2. Define base prompt
    base_prompt = """You are a legal assistant specialized in contract law.
    Provide accurate information with proper citations.
    Always remind users to consult with a licensed attorney for legal advice."""
    
    # 3. Initialize evolution system
    system = PromptEvolution(config_path="config.yaml")
    
    # 4. Evolve the prompt
    final_state = system.evolve_prompt(
        base_prompt=base_prompt,
        test_scenarios=test_scenarios
    )
    
    # 5. Save the evolved prompt
    output_file = f"prompts/legal_contract_evolved_{final_state.generation}.txt"
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(final_state.current_prompt)
    
    print(f"✅ Evolved prompt saved to: {output_file}")
    print(f"📊 Success rate: {final_state.results.success_rate:.1%}")
    print(f"⭐ Quality score: {final_state.results.quality_score:.2f}")

if __name__ == "__main__":
    main()
```

---

## 7. Summary

| **Component** | **Purpose** |
|--------------|-------------|
| **TestScenario** | Defines what to test (input, expected output, bad example) |
| **execute_prompt()** | Runs the prompt against all scenarios |
| **_judge_response()** | Uses LLM to evaluate if response is good |
| **evaluate_results()** | Calculates metrics (success rate, quality score) |
| **evolve_prompt()** | Iteratively improves the prompt based on test results |

**The France and Quantum examples are hardcoded demos.** In production, you would:
1. Create domain-specific test scenarios (legal, medical, support, etc.)
2. Save them as JSON files in `data/test_scenarios/`
3. Load them dynamically based on your domain
4. Use Task Definer to auto-generate scenarios from task descriptions

Would you like me to create a scenario generator that connects Task Definer → Test Scenarios?
