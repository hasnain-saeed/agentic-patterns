from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def create_completion(client: str, model: str, messages: str):
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def create_prompt_struct(content: str, role: str, tag: str = '') -> dict:
    if tag:
        content = f"<{tag}>{content}</{tag}>"
    return {'role': role, 'content': content}
