"""
Prompt building module - Constructs different types of prompts for LLMs
"""
from typing import List, Dict


class PromptBuilder:
    
    @staticmethod
    def standard_prompt(question: str) -> str:
        """Build standard prompt - direct answer without reasoning"""
        return f"Question: {question}\n\nAnswer:"
    
    @staticmethod
    def zero_shot_cot(question: str) -> str:
        """Build zero-shot Chain-of-Thought prompt"""
        return f"Question: {question}\n\nLet's think step by step."
    
    @staticmethod
    def few_shot_cot(question: str, examples: List[Dict] = None) -> str:
        """Build few-shot Chain-of-Thought prompt with examples"""
        if examples is None:
            examples = [
                {
                    "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
                    "reasoning": "Roger starts with 5 balls. He buys 2 cans, each with 3 balls, so that's 2 × 3 = 6 more balls. Total: 5 + 6 = 11.",
                    "answer": "11"
                },
                {
                    "question": "A bakery sells cookies in boxes of 12. If a customer buys 4 boxes, how many cookies do they have?",
                    "reasoning": "Each box has 12 cookies. Buying 4 boxes means 4 × 12 = 48 cookies.",
                    "answer": "48"
                }
            ]
        
        few_shot_text = "Here are some examples:\n\n"
        for ex in examples:
            few_shot_text += f"Question: {ex['question']}\n"
            few_shot_text += f"Reasoning: {ex['reasoning']}\n"
            few_shot_text += f"Answer: {ex['answer']}\n\n"
        
        few_shot_text += f"Now answer the following question similarly:\n"
        few_shot_text += f"Question: {question}\n"
        few_shot_text += f"Reasoning:"
        
        return few_shot_text
    
    @staticmethod
    def get_prompt(prompt_type: str, question: str) -> str:
        """Unified interface to get prompt by type"""
        if prompt_type == "standard":
            return PromptBuilder.standard_prompt(question)
        elif prompt_type == "zero_shot_cot":
            return PromptBuilder.zero_shot_cot(question)
        elif prompt_type == "few_shot_cot":
            return PromptBuilder.few_shot_cot(question)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")