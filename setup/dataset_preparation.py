"""
Advanced dataset preparation for fine-tuning Ollama models with coding expertise.
Focuses on reasoning datasets optimized for software development workflows.
"""

import json
import asyncio
import aiofiles
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import datasets
from transformers import AutoTokenizer

@dataclass
class CodeReasoningExample:
    """Structure for code reasoning training examples."""
    instruction: str
    input_code: str
    reasoning_steps: List[str]
    output_code: str
    language: str
    complexity: str
    tags: List[str]

class AdvancedCodeDatasetBuilder:
    """Build high-quality reasoning datasets for code fine-tuning."""
    
    def __init__(self, tokenizer_name: str = "microsoft/CodeBERT-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.reasoning_templates = self._load_reasoning_templates()
    
    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Templates for structured reasoning in code tasks."""
        return {
            "debug": """I need to debug this {language} code:

{code}

Let me analyze this step by step:
1. **Understanding the problem**: {problem_analysis}
2. **Identifying potential issues**: {issue_identification}
3. **Root cause analysis**: {root_cause}
4. **Solution strategy**: {solution_strategy}
5. **Implementation**: {implementation}

Here's the corrected code:

```{language}
{fixed_code}
```""",
            
            "optimization": """I need to optimize this {language} code for better performance:

{code}

My optimization approach:
1. **Performance analysis**: {performance_analysis}
2. **Bottleneck identification**: {bottlenecks}
3. **Optimization strategies**: {strategies}
4. **Trade-off considerations**: {tradeoffs}
5. **Implementation**: {implementation}

Optimized code:

```{language}
{optimized_code}
```""",
            
            "refactoring": """I need to refactor this {language} code for better maintainability:

{code}

Refactoring process:
1. **Code smell analysis**: {code_smells}
2. **Design pattern opportunities**: {patterns}
3. **Separation of concerns**: {separation}
4. **Clean code principles**: {principles}
5. **Implementation**: {implementation}

Refactored code:

```{language}
{refactored_code}
```""",
            
            "feature_addition": """I need to add a new feature to this {language} code:

{code}

Feature: {feature_description}

Development approach:
1. **Requirements analysis**: {requirements}
2. **Design considerations**: {design}
3. **Integration strategy**: {integration}
4. **Testing approach**: {testing}
5. **Implementation**: {implementation}

Enhanced code:

```{language}
{enhanced_code}
```"""
        }
    
    async def create_reasoning_dataset(self) -> List[Dict[str, Any]]:
        """Create comprehensive reasoning dataset for code tasks."""
        
        # Load base datasets
        datasets_config = [
            {
                "name": "codeparrot/github-code-clean",
                "split": "train[:10000]",
                "languages": ["python", "javascript", "typescript", "go", "rust"]
            },
            {
                "name": "bigcode/the-stack-dedup", 
                "split": "train[:5000]",
                "languages": ["python", "javascript", "java", "cpp"]
            },
            {
                "name": "openai_humaneval",
                "split": "test",
                "languages": ["python"]
            }
        ]
        
        reasoning_examples = []
        
        for config in datasets_config:
            dataset = datasets.load_dataset(config["name"], split=config["split"])
            examples = await self._process_dataset_for_reasoning(dataset, config["languages"])
            reasoning_examples.extend(examples)
        
        # Add synthetic reasoning examples
        synthetic_examples = await self._generate_synthetic_reasoning_examples()
        reasoning_examples.extend(synthetic_examples)
        
        return reasoning_examples
    
    async def _process_dataset_for_reasoning(self, dataset, languages: List[str]) -> List[Dict[str, Any]]:
        """Process existing datasets to add reasoning components."""
        examples = []
        
        for item in dataset:
            if item.get("language") in languages:
                # Create multiple reasoning tasks from each code sample
                reasoning_tasks = [
                    await self._create_debug_task(item),
                    await self._create_optimization_task(item),
                    await self._create_refactoring_task(item),
                    await self._create_explanation_task(item)
                ]
                examples.extend([t for t in reasoning_tasks if t])
        
        return examples
    
    async def _create_debug_task(self, code_item: Dict) -> Optional[Dict[str, Any]]:
        """Create a debugging reasoning task."""
        code = code_item.get("content", "")
        language = code_item.get("language", "python")
        
        if len(code.split('\n')) < 5:  # Skip very short code
            return None
        
        # Introduce a realistic bug
        buggy_code, bug_description = self._introduce_bug(code, language)
        
        return {
            "instruction": f"Debug the following {language} code that has a {bug_description}:",
            "input": buggy_code,
            "output": self.reasoning_templates["debug"].format(
                language=language,
                code=buggy_code,
                problem_analysis=f"The code appears to have a {bug_description}",
                issue_identification="Analyzing the logic flow and potential edge cases",
                root_cause=f"The issue stems from {bug_description}",
                solution_strategy="Apply defensive programming and proper error handling",
                implementation="Fix the identified issue while maintaining code quality",
                fixed_code=code
            ),
            "reasoning_type": "debugging",
            "language": language,
            "complexity": "intermediate"
        }
    
    def _introduce_bug(self, code: str, language: str) -> tuple[str, str]:
        """Introduce realistic bugs for debugging practice."""
        bug_patterns = {
            "python": [
                ("== None", "is None", "equality check bug"),
                ("range(len(", "enumerate(", "inefficient iteration"),
                ("except:", "except Exception:", "broad exception handling")
            ],
            "javascript": [
                ("== null", "=== null", "loose equality bug"),
                ("var ", "const ", "variable declaration issue"),
                ("parseInt(", "Number(", "type conversion issue")
            ]
        }
        
        patterns = bug_patterns.get(language, bug_patterns["python"])
        for wrong, correct, description in patterns:
            if correct in code:
                buggy_code = code.replace(correct, wrong, 1)
                return buggy_code, description
        
        return code, "logic error"

class ReasoningDatasetConfig:
    """Configuration for reasoning dataset creation."""
    
    # Advanced reasoning patterns for code
    REASONING_PATTERNS = {
        "chain_of_thought": {
            "template": "Let me think through this step by step:\n{steps}",
            "steps_format": "Step {n}: {description}"
        },
        "problem_decomposition": {
            "template": "I'll break this problem down:\n{components}",
            "components_format": "- {component}: {explanation}"
        },
        "analogical_reasoning": {
            "template": "This is similar to {analogy}, so I can apply {principle}:\n{application}",
        },
        "constraint_satisfaction": {
            "template": "Given the constraints:\n{constraints}\nI need to find a solution that:\n{solution_criteria}"
        }
    }
    
    # Code quality metrics to optimize for
    QUALITY_METRICS = {
        "readability": ["variable naming", "code structure", "documentation"],
        "maintainability": ["modularity", "separation of concerns", "DRY principle"],
        "performance": ["time complexity", "space complexity", "algorithmic efficiency"],
        "security": ["input validation", "error handling", "secure practices"],
        "testability": ["unit testable", "dependency injection", "pure functions"]
    }