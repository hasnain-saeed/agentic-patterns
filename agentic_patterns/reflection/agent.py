import structlog
from datetime import datetime
from pathlib import Path
from enum import Enum
from colorama import Fore
from dotenv import load_dotenv
from openai import OpenAI

import prompts
from agentic_patterns.utils import create_completion, create_prompt_struct
from agentic_patterns.history import ChatHistory

logger = structlog.stdlib.get_logger(__name__)
load_dotenv()

LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class ReflectionAgent:
    class CompletionColor(Enum):
        DEFAULT = Fore.WHITE
        REFLECTION = Fore.BLUE
        GENERATION = Fore.GREEN

    def __init__(self, model='gpt-4o-mini', save_logs=False):
        self.client = OpenAI()
        self.model = model
        self.save_logs = save_logs
        self.logs = []

    def _generate_completion(self, history: list, completion_color: CompletionColor = CompletionColor.DEFAULT) -> str:
        response = create_completion(self.client, self.model, history)
        if self.save_logs:
            self.logs.append(f'\n\n{completion_color.name}\n\n' + response)
        else:
            print(completion_color.value, f'\n\n{completion_color.name}\n\n', response)
        return response

    def generate(self, generation_history: list) -> str:
        return self._generate_completion(generation_history, completion_color=self.CompletionColor.GENERATION)

    def reflect(self, reflection_history: list) -> str:
        return self._generate_completion(reflection_history, completion_color=self.CompletionColor.REFLECTION)

    def run(self, user_message: str, steps: int = 5, generation_prompt: str = '', reflection_prompt: str = '') -> str:
        generation_prompt = f'{generation_prompt}\n{prompts.BASE_GENERATION_SYSTEM_PROMPT}'
        reflection_prompt = f'{reflection_prompt}\n{prompts.BASE_REFLECTION_SYSTEM_PROMPT}'

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rollout_{timestamp}.log"
            log_path = LOGS_DIR / filename
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("".join(self.logs))

            logger.info("rollout_log_saved", path=str(log_path))
            return log_path

        return completion
