"""
Script to replace OpenAI-specific API calls with unified call_llm function
"""
import re

# Read the file
with open('prompt_evolution.py', 'r') as f:
    content = f.read()

# Pattern to match the OpenAI API call
pattern = r'response = self\.client\.chat\.completions\.create\(\s*model=self\.model,\s*messages=(\[[\s\S]*?\]),\s*temperature=self\.temperature(?:,\s*max_tokens=self\.config\[\'models\'\]\[\'[^\']+\'\]\[\'max_tokens\'\])?\s*\)\s*return response\.choices\[0\]\.message\.content'

# Replacement
replacement = r'return call_llm(\n                self.client,\n                self.provider,\n                messages=\1,\n                model=self.model,\n                temperature=self.temperature,\n                max_tokens=self.max_tokens\n            )'

# Perform replacement
content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Also handle cases where the response is used later
pattern2 = r'(\w+_response|\w+) = self\.client\.chat\.completions\.create\(\s*model=self\.model,\s*messages=(\[[\s\S]*?\]),\s*temperature=self\.temperature(?:,\s*max_tokens=self\.config\[\'models\'\]\[\'[^\']+\'\]\[\'max_tokens\'\])?\s*\)'

def replace_func(match):
    var_name = match.group(1)
    messages = match.group(2)
    return f'''{var_name} = call_llm(
                self.client,
                self.provider,
                messages={messages},
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )'''

content = re.sub(pattern2, replace_func, content, flags=re.MULTILINE)

# Write back
with open('prompt_evolution.py', 'w') as f:
    f.write(content)

print("Replaced OpenAI API calls with unified call_llm function")
