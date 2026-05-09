"""
Model client module - Wraps Azure OpenAI API calls
Handles retries, rate limiting, and error recovery
"""
import time
from typing import Optional, List
from src.config import Config


class ModelClient:
    """
    Client for interacting with Azure OpenAI models
    
    Features:
    - Automatic retry on failure
    - Configurable temperature and max tokens
    - Error handling with exponential backoff
    """
    
    def __init__(self, retry_times: int = 3, delay: float = 1.0):
        """
        Initialize model client
        
        Args:
            retry_times: Number of retry attempts on failure
            delay: Initial delay between retries (exponential backoff)
        """
        self.client = Config.get_client()
        self.deployment = Config.AZURE_DEPLOYMENT
        self.max_tokens = Config.MAX_TOKENS
        self.temperature = Config.TEMPERATURE
        self.retry_times = retry_times
        self.delay = delay
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate response from the model
        
        Args:
            prompt: Input prompt string
            max_tokens: Maximum tokens in response (uses default if None)
            
        Returns:
            Generated response text
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        for attempt in range(self.retry_times):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that provides clear, step-by-step reasoning."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.retry_times} failed: {e}")
                if attempt < self.retry_times - 1:
                    # Exponential backoff
                    wait_time = self.delay * (2 ** attempt)
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise e
        
        return ""
    
    def generate_batch(self, prompts: List[str], show_progress: bool = True) -> List[str]:
        """
        Generate responses for a batch of prompts
        
        Args:
            prompts: List of prompt strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of response strings
        """
        responses = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(prompts, desc="Generating responses")
        else:
            iterator = prompts
        
        for prompt in iterator:
            responses.append(self.generate(prompt))
        
        return responses