# Mutation Feedback System

## Your Questions Answered:

### Q: During mutation, do we provide why the prompt scored less?
**A: YES! (Now we do)**

### Q: Do we give other prompts to it?
**A: YES! We show the best prompt for comparison**

---

## What Feedback Is Provided to Mutations

When generating a new mutation from a parent prompt, the system now provides:

### 1. **Current Score**
```
Current score: 0.72
```
Tells the mutation system how well this prompt is performing.

### 2. **Weaknesses** (from Judge)
```
Weaknesses:
- Lacks specificity
- Missing output format instructions
- Unclear about edge cases
```
Exact feedback from the judge on what's wrong.

### 3. **Suggestions** (from Judge)
```
Suggestions:
- Add concrete examples
- Specify desired output format
- Include error handling instructions
```
Actionable improvements the judge recommends.

### 4. **Best Prompt Comparison** (if not the best)
```
Best prompt score: 0.88
Best prompt: "Analyze customer feedback by categorizing sentiment (positive/negative/neutral) and extracting top 3 actionable insights with supporting quotes for product improvement."
```

Shows what the current best prompt looks like, so the mutation can learn from it!

---

## Example: Full Feedback Passed to Mutation

```
Current score: 0.72

Weaknesses:
- Too generic
- Missing domain-specific terms
- No output format specified

Suggestions:
- Add medical terminology
- Specify structured output format
- Include example of desired summary

Best prompt score: 0.85
Best prompt: "Summarize patient discharge notes including diagnosis, medications, follow-up instructions, and pending test results in bullet-point format for handoff to next provider."
```

The mutation system receives ALL of this and uses it to improve the prompt!

---

## How It Works in Practice

### Generation 1:
- **No feedback** - initial mutations are blind
- Just use domain context

### Generation 2+:
Each parent prompt provides:
- Its own weaknesses and suggestions
- Comparison to the best prompt in the population
- Specific guidance on what to improve

### Example Flow:

**Parent Prompt A (score: 0.72)**
```
"Summarize text"
```

**Feedback to Mutation System:**
```
Current score: 0.72

Weaknesses:
- Lacks medical specificity
- No output structure defined

Suggestions:
- Add clinical terminology
- Specify output format

Best prompt score: 0.85
Best prompt: "Summarize patient discharge notes including diagnosis, medications, follow-up..."
```

**Mutation System Generates:**
```
"Summarize patient medical records including diagnosis, treatment plan, medications, and follow-up instructions in structured bullet-point format."
```

**New Score:** 0.82 ✅ (improved by learning from feedback!)

---

## Benefits

### 1. **Faster Convergence**
- Mutations know exactly what to fix
- Don't repeat mistakes
- Learn from the best prompt

### 2. **Smarter Mutations**
- Address specific weaknesses
- Follow judge's suggestions
- Incorporate successful patterns from best prompt

### 3. **Less Random**
- Not just random changes
- Targeted improvements
- Evidence-based mutations

---

## When Feedback Is Used

### Initial Population (Generation 1):
❌ **No feedback** - prompts haven't been evaluated yet
- Uses only domain/context

### Subsequent Generations:
✅ **Full feedback** - prompts have been evaluated
- Uses scores, weaknesses, suggestions, best prompt

### If Target Hit Early:
⚠️ **Might not use feedback** - optimization stops
- Like in the code example: hit 0.91 in Gen 1!

---

## Technical Details

### Feedback Structure:
```python
linguistic_feedback = f"""
Current score: {parent.fitness_score:.2f}

Weaknesses: {'; '.join(parent.metadata['weaknesses'])}

Suggestions: {'; '.join(parent.metadata['suggestions'])}

Best prompt score: {best_member.fitness_score:.2f}
Best prompt: {best_member.prompt_text}
"""
```

### Passed to Mutation System:
```python
mutation = self.challenger.mutate(
    prompt=parent.prompt_text,
    linguistic_feedback=linguistic_feedback  # Here!
)
```

### Used in Mutation Prompt:
```
EVALUATION FEEDBACK:
Current score: 0.72
Weaknesses: Lacks specificity
Suggestions: Add concrete examples
Best prompt score: 0.85
Best prompt: "..."

Make sure the improved prompt addresses the weaknesses...
```

---

## Summary

**YES**, the system now provides:
1. ✅ Why the prompt scored less (weaknesses)
2. ✅ What to improve (suggestions)
3. ✅ The best prompt for comparison
4. ✅ Current vs best score

This makes mutations **intelligent** and **targeted** rather than random!
