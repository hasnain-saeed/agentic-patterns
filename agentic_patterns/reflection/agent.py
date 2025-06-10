import structlog
from datetime import datetime
from pathlib import Path
from enum import Enum
from colorama import Fore
from dotenv import load_dotenv
from openai import OpenAI

from agentic_patterns.utils import create_completion, create_prompt_struct
from agentic_patterns.history import ChatHistory
from .prompts import BASE_GENERATION_SYSTEM_PROMPT, BASE_REFLECTION_SYSTEM_PROMPT

logger = structlog.stdlib.get_logger(__name__)
load_dotenv()

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class CompletionColor(Enum):
    DEFAULT = Fore.WHITE
    REFLECTION = Fore.BLUE
    GENERATION = Fore.GREEN

class ReflectionAgent:
    def __init__(self, model='gpt-4o-mini', run_local: bool=False, save_logs: str=False):
        self.model = model
        self.save_logs = save_logs
        self.logs = []
        self.client = self._initialize_client(run_local)

    def _initialize_client(self, run_local: bool):
        if run_local:
            return OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama',
            )
        return OpenAI()

    def _generate_completion(self, history: list, completion_color: CompletionColor = CompletionColor.DEFAULT) -> str:
        response = create_completion(self.client, self.model, history)
        if self.save_logs:
            self.logs.append(f'\n\n{completion_color.name}\n\n' + response)
        else:
            print(completion_color.value, f'\n\n{completion_color.name}\n\n', response)
        return response

    def _save_logs(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model}_{timestamp}.log"
        log_path = LOGS_DIR / filename
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("".join(self.logs))

        logger.info("Logs saved:", path=str(log_path))

    def generate(self, generation_history: list) -> str:
        return self._generate_completion(generation_history, completion_color=CompletionColor.GENERATION)

    def reflect(self, reflection_history: list) -> str:
        return self._generate_completion(reflection_history, completion_color=CompletionColor.REFLECTION)

    def run(self, user_message: str, steps: int = 5, generation_prompt: str = '', reflection_prompt: str = '') -> str:
        generation_prompt = f'{generation_prompt}\n{BASE_GENERATION_SYSTEM_PROMPT}'
        reflection_prompt = f'{reflection_prompt}\n{BASE_REFLECTION_SYSTEM_PROMPT}'

        generation_history = ChatHistory(
            messages=[
                create_prompt_struct(content=generation_prompt, role='system'),
                create_prompt_struct(content=user_message, role='user'),
            ],
            total_length=3,
            popout_index=1, # keep the system prompt and discard remaining responses
        )

        reflection_history = ChatHistory(
            messages=[
                create_prompt_struct(content=reflection_prompt, role='system'),
            ],
            total_length=3,
            popout_index=1,
        )

        for _ in range(steps):
            completion = self.generate(generation_history)
            generation_history.append(create_prompt_struct(content=completion, role='assistant'))
            reflection_history.append(create_prompt_struct(content=completion, role='user'))

            reflection = self.reflect(reflection_history)
            if "<OK>" in reflection:
                if self.save_logs:
                    self.logs.append('\n\nStop Sequence found. Stopping the reflection loop ... \n\n')
                else:
                    print(Fore.RED, '\n\nStop Sequence found. Stopping the reflection loop ... \n\n',)
                break

            generation_history.append(create_prompt_struct(content=reflection, role='user'))
            reflection_history.append(create_prompt_struct(content=reflection, role='assistant'))

        if self.save_logs:
            self._save_logs()

        return completion
