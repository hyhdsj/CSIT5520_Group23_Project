"""
Reasoning analysis module - Analyzes the quality of model reasoning
Counts reasoning steps and checks for reasoning structure
"""
import re
from typing import Dict, List


class ReasoningAnalyzer:
    """
    Analyze reasoning patterns in model responses
    
    Features:
    - Count number of reasoning steps
    - Detect reasoning keywords (first, then, therefore, etc.)
    - Check for calculations and conclusions
    """
    
    @staticmethod
    def count_reasoning_steps(response: str) -> int:
        """
        Count the number of reasoning steps in a response
        
        Looks for:
        - Step keywords (first, second, then, next, finally)
        - Numbered lists (1., 2., 3.)
        - Multiple numbers (indicating calculations)
        
        Args:
            response: Model response text
            
        Returns:
            Estimated number of reasoning steps
        """
        step_indicators = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bthen\b',
            r'\bnext\b', r'\bfinally\b', r'\blastly\b',
            r'^\d+\.',  # Numbered list at line start
            r'\bstep\s*\d+\b'
        ]
        
        count = 0
        for pattern in step_indicators:
            count += len(re.findall(pattern, response, re.IGNORECASE | re.MULTILINE))
        
        # If no explicit steps, check for multiple numbers (calculations)
        if count == 0:
            numbers = re.findall(r'\d+', response)
            if len(numbers) > 2:
                count = len(numbers) - 1
        
        return max(1, count) if count > 0 else 0
    
    @staticmethod
    def has_reasoning_structure(response: str) -> Dict[str, bool]:
        """
        Check if response has proper reasoning structure
        
        Args:
            response: Model response text
            
        Returns:
            Dictionary with boolean flags for different structures
        """
        return {
            "has_step_keywords": bool(re.search(
                r'\b(step|then|first|next|finally|because|therefore)\b', 
                response, re.IGNORECASE
            )),
            "has_calculation": bool(re.search(
                r'\d+\s*[+\-*/]\s*\d+', response
            )),
            "has_conclusion": bool(re.search(
                r'\b(so|therefore|thus|hence|conclusion|answer:)\b', 
                response, re.IGNORECASE
            )),
            "is_coherent": len(response.split()) > 20  # Rough heuristic
        }
    
    @staticmethod
    def analyze_reasoning_quality(responses: List[str]) -> Dict:
        """
        Analyze reasoning quality across multiple responses
        
        Args:
            responses: List of model response texts
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not responses:
            return {"avg_steps": 0, "structure_stats": {}, "samples_analyzed": 0}
        
        total_steps = 0
        structure_counts = {
            "has_step_keywords": 0,
            "has_calculation": 0,
            "has_conclusion": 0,
            "is_coherent": 0
        }
        
        for response in responses:
            steps = ReasoningAnalyzer.count_reasoning_steps(response)
            total_steps += steps
            
            structure = ReasoningAnalyzer.has_reasoning_structure(response)
            for key, value in structure.items():
                if value:
                    structure_counts[key] += 1
        
        return {
            "avg_steps": total_steps / len(responses),
            "structure_stats": {
                key: count / len(responses) 
                for key, count in structure_counts.items()
            },
            "samples_analyzed": len(responses)
        }