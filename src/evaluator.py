"""
Evaluation module - Calculates metrics for model predictions
"""
import re
import string
from typing import Tuple, List, Dict


class Evaluator:
    
    @staticmethod
    def extract_answer_from_response(response: str, choices: List[str] = None) -> str:
        """
        Extract the final answer from model response.
        
        Handles:
        - Math: numbers, "#### 72" format
        - Commonsense: "**C. knowing more**", "C. knowing more", "Answer: knowing more"
        - Multihop: direct text answers
        """
        if not response:
            return ""
        
        # Work on a copy
        text = response.strip()
        
        # ========== STEP 1: Remove Markdown formatting ==========
        # Remove **bold**
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        # Remove *italic*
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # ========== STEP 2: Try to extract answer ==========
        
        # Pattern 1: "#### 72" (GSM8K math format)
        match = re.search(r'####\s*(.+?)(?:\n|$)', text)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: "Answer: C. knowing more" or "Answer: knowing more"
        match = re.search(r'(?:answer|Answer|ANSWER)\s*[:;]\s*(.+?)(?:\n|$)', text)
        if match:
            ans = match.group(1).strip().rstrip('.,!?')
            # Remove letter prefix like "C. " or "C "
            ans = re.sub(r'^[A-E][\.\s]*', '', ans)
            if ans:
                return ans
        
        # Pattern 3: "**C. knowing more**" or "C. knowing more" (Commonsense format)
        # This pattern looks for a letter (A-E) followed by dot and text
        match = re.search(r'([A-E])\.\s+([A-Za-z\s]+?)(?:\n|$|\.\s|,|\))', text, re.IGNORECASE)
        if match:
            ans = match.group(2).strip().rstrip('.,!?)')
            if ans and len(ans) > 1:
                return ans
        
        # Pattern 4: Look for the last line that contains meaningful text (for multihop)
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            # Skip lines that are just a single letter
            if re.match(r'^[A-E]$', line, re.IGNORECASE):
                continue
            # Skip lines that are just "Answer:" or similar
            if re.match(r'^(answer|Answer|Reasoning):?\s*$', line):
                continue
            # Remove any leading letter prefix
            line = re.sub(r'^[A-E][\.\s]*', '', line)
            # If line has reasonable length, return it
            if len(line) > 2 and len(line) < 200:
                return line.strip()
        
        # Pattern 5: Last number (for math problems without ####)
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        # Pattern 6: Fallback - return first meaningful line
        for line in text.split('\n'):
            line = line.strip()
            if len(line) > 5 and len(line) < 150:
                return line
        
        return text[:100].strip()
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """Normalize answer string for comparison"""
        if not s:
            return ""
        
        s = str(s).lower().strip()
        # Remove punctuation
        s = re.sub(r'[^\w\s]', ' ', s)
        # Remove extra spaces
        s = ' '.join(s.split())
        return s
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> bool:
        """Calculate Exact Match score"""
        if not prediction or not ground_truth:
            return False
        
        norm_pred = Evaluator.normalize_answer(prediction)
        norm_gt = Evaluator.normalize_answer(ground_truth)
        
        # Direct comparison
        if norm_pred == norm_gt:
            return True
        
        # Check if one contains the other (for partial matches)
        if norm_pred in norm_gt or norm_gt in norm_pred:
            return True
        
        return False
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """Calculate token-level F1 score"""
        norm_pred = Evaluator.normalize_answer(prediction)
        norm_gt = Evaluator.normalize_answer(ground_truth)
        
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def evaluate_response(response: str, ground_truth: str, choices: List[str] = None) -> Tuple[float, bool]:
        """Evaluate a single response"""
        extracted = Evaluator.extract_answer_from_response(response, choices)
        em = Evaluator.exact_match(extracted, ground_truth)
        f1 = Evaluator.f1_score(extracted, ground_truth)
        return f1, em
    
    @staticmethod
    def calculate_metrics(results: List[Dict]) -> Dict:
        """Calculate aggregate metrics"""
        if not results:
            return {"exact_match": 0.0, "f1_score": 0.0, "total_samples": 0}
        
        total = len(results)
        em_correct = sum(1 for r in results if r.get("exact_match", False))
        total_f1 = sum(r.get("f1_score", 0.0) for r in results)
        
        return {
            "exact_match": em_correct / total,
            "f1_score": total_f1 / total,
            "total_samples": total
        }