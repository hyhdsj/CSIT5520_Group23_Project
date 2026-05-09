import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

class Config:
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-10-21")
    
    # Experiment Parameters
    MAX_TOKENS = 512
    TEMPERATURE = 0  # Ensure reproducibility
    SAMPLE_SIZE = 50  # Number of samples per dataset

    @staticmethod
    def get_client():
        """Get the Azure OpenAI client"""
        return AzureOpenAI(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION
        )

    @staticmethod
    def validate():
        """
        Validate that required configuration is present
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        if not Config.AZURE_API_KEY:
            print("ERROR: Please set AZURE_OPENAI_API_KEY in .env file")
            return False
        if not Config.AZURE_ENDPOINT:
            print("ERROR: Please set AZURE_OPENAI_ENDPOINT in .env file")
            return False
        if not Config.AZURE_DEPLOYMENT:
            print("ERROR: Please set AZURE_OPENAI_DEPLOYMENT in .env file")
            return False
        
        # Optional: Print configuration status (without exposing full key)
        print(f"✓ Azure endpoint: {Config.AZURE_ENDPOINT}")
        print(f"✓ Deployment: {Config.AZURE_DEPLOYMENT}")
        print(f"✓ API key: {'*' * 10}{Config.AZURE_API_KEY[-4:] if Config.AZURE_API_KEY else 'NOT SET'}")
        
        return True