# Task Definer System

The **Task Definer System** is a Python-based framework that leverages Large Language Models (LLMs) to analyze user prompts, extract context, define problems, classify tasks, validate definitions, and generate actionable task specifications. It supports user profiling, chat history tracking, and automated batch task processing.

---

## 📂 Modules & Imports

* **Standard Libraries**: `json`, `os`, `datetime`, `hashlib`, `logging`
* **Typing**: `List`, `Dict`, `Any`, `Optional`
* **Dataclasses**: `dataclass`, `asdict`
* **Enums**: `Enum`
* **LLM API**: `openai`, `OpenAI`

---

## 🏷️ Enumerations

### `OutputType`

Defines types of task outputs:

* `TEXT`
* `CODE`
* `DATA`
* `ANALYSIS`
* `DESIGN`
* `PLAN`
* `DOCUMENTATION`
* `API`
* `UI_COMPONENT`
* `OTHER`

### `Priority`

Defines task priority levels:

* `LOW`
* `MEDIUM`
* `HIGH`
* `CRITICAL`

### `Complexity`

Defines task complexity levels:

* `SIMPLE`
* `MODERATE`
* `COMPLEX`
* `HIGHLY_COMPLEX`

---

## 👤 UserProfile (Dataclass)

Represents a user profile with:

* `user_id` (str) – Unique user identifier
* `name` (Optional[str]) – User’s name
* `domain_expertise` (List[str]) – Areas of expertise
* `preferred_output_formats` (List[str]) – Preferred output formats
* `typical_complexity` (str) – Usual task complexity
* `past_projects` (List[str]) – Past project summaries
* `constraints_history` (List[str]) – Common constraints
* `communication_style` (str) – Professional/technical style
* `technical_level` (str) – Beginner, intermediate, expert

🔹 **Method**:
`to_context_string()` – Converts profile into a context string for the LLM.

---

## ⚙️ LLMTaskDefinitionSystem

Main system that prepares structured task definitions.

### **Initialization**

```python
LLMTaskDefinitionSystem(api_key: str = None, model: str = "gpt-4-turbo-preview")
```

* Uses OpenAI API (key from parameter or `OPENAI_API_KEY` environment variable).
* Stores chat history and user profiles in-memory (replaceable with a database).

---

### **Core Methods**

1. **User Profile & History**

   * `load_user_profile(user_id)` → Returns a user profile
   * `get_chat_history(user_id, limit=10)` → Gets recent messages
   * `save_chat_message(user_id, role, content)` → Saves a message

2. **Context Extraction**

   * `extract_context_with_priority(raw_prompt, chat_history, user_profile)`
     Uses LLM to analyze prompt, chat history, and profile to extract context.

3. **Problem Definition**

   * `generate_problem_statement(raw_prompt, context)`
     Produces a clear problem statement, objectives, and success criteria.

4. **Task Classification**

   * `classify_task(raw_prompt, problem_def, context)`
     Classifies task metadata: type, priority, complexity, skills, dependencies, risk level.

5. **Validation**

   * `validate_and_enhance(task_definition)`
     Validates the task definition for completeness, clarity, and execution readiness.

6. **Task Preparation**

   * `prepare_task_definition(raw_prompt, user_id="default")`
     End-to-end task processing pipeline:

     * Saves history
     * Extracts context
     * Generates problem definition
     * Classifies task
     * Validates & enhances
     * Returns finalized JSON task definition

7. **Persistence**

   * `save_to_file(task_definition, filename=None)`
     Saves definition as a JSON file.

8. **Batch Processing**

   * `batch_process(prompts, user_id="default")`
     Processes multiple prompts at once.

---

## 📋 ProfileManager

Utility class to update user profiles based on completed task definitions:

* `update_profile_from_task(user_id, task_definition)`

---

## 🚀 Example Usage

```python
def main():
    # Initialize system
    system = LLMTaskDefinitionSystem(api_key="OPENAI_API_KEY")

    # Create and store a sample profile
    profile = UserProfile(
        user_id="user123",
        name="John Developer",
        domain_expertise=["software development", "web apps"],
        preferred_output_formats=["json", "python", "react"],
        typical_complexity="complex",
        past_projects=["Build REST API", "Data Dashboard"],
        constraints_history=["scalable", "SOLID principles"],
        communication_style="technical",
        technical_level="expert"
    )
    system.user_profiles["user123"] = profile

    # Add sample chat history
    system.chat_history_store["user123"] = [
        {"role": "user", "content": "I need to build a microservice", "timestamp": "2024-01-01T10:00:00"}
    ]

    # Process a new task
    prompt = "Create a real-time analytics dashboard for IoT sensors."
    task_def = system.prepare_task_definition(prompt, "user123")

    # Save task definition
    filename = system.save_to_file(task_def)

    # Batch process multiple prompts
    batch_prompts = ["Implement OAuth2", "Create CI/CD testing pipeline"]
    results = system.batch_process(batch_prompts, "user123")
```

---

## ✅ Features

* **LLM-powered context extraction**
* **Problem statement generation**
* **Task classification (priority, complexity, effort, skills)**
* **Validation & enhancement of definitions**
* **User profiling & adaptive learning**
* **Batch task processing**
* **JSON export**

---

## 📄 Output Example (Simplified)

```json
{
  "id": "TASK-20251004123456-abc12345",
  "created_at": "2025-10-04T12:34:56",
  "user_id": "user123",
  "status": "finalized",
  "problem_statement": "Build a real-time IoT analytics dashboard",
  "objectives": ["Process streaming data", "Display key metrics", "Send alerts"],
  "success_criteria": ["Threshold alerts working", "Metrics updated in real-time"],
  "classification": {
    "output_type": "ui_component",
    "priority": "high",
    "complexity": "complex",
    "risk_level": "medium"
  },
  "validation": {
    "validation_status": "valid",
    "ready_for_execution": true
  }
}
```

---

## 📌 Notes

* Replace `"OPENAI_API_KEY"` with a valid OpenAI key.
* In production, replace in-memory history & profile storage with a database.
* Works best with `gpt-4-turbo-preview` but supports other models.

---
