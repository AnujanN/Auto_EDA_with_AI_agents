import os
from pathlib import Path
from crewai import LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    def __init__(self):
        # Get the project root directory
        self.project_root = Path(__file__).parent
        
        # Configuration file paths
        self.agents_path = self.project_root / "EDA_config" / "agents.yaml"
        self.tasks_path = self.project_root / "EDA_config" / "tasks.yaml"
        
        # Initialize LLM
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Google Gemini LLM using CrewAI's LLM class"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Use the same pattern as your working config
        return LLM(
            model="gemini/gemini-2.5-flash",
            api_key=api_key
        )

# Create global config instance
config = Config()
