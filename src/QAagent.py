from dotenv import load_dotenv
import os
import time
from rich.console import Console
from together import Together

console = Console()
load_dotenv()
# Load environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=TOGETHER_API_KEY  if TOGETHER_API_KEY else None)


class QAAgent:
    def __init__(self):
        self.console = console
        self.together_api_key = TOGETHER_API_KEY

    def 