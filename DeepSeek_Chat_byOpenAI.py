# Позволяет вести диалог с DeepSeek
# https://api-docs.deepseek.com/guides/reasoning_model
# https://api-docs.deepseek.com/api/create-chat-completion

from openai import OpenAI, APITimeoutError
import httpx

DEEPSEEK_API_KEY = "INSERT_YOUR_API_KEY_FROM_DeepSeek_ACCOUNT"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
RESPONSE_STRIP_CHARS = '«»„““”"❝❞„⹂〝〞〟＂‹›❮❯‚‘‘‛’❛❜❟`\'., '
DEEPSEEK_MODELS = {
    'deepseek': "deepseek-chat",
    'deepseek-reason': "deepseek-reasoner",
}


class DeepSeek:
    """DeepSeek chat assistant"""

    def __init__(
            self,
            api_key: str = '',  # DeepSeek Api-key
            proxy: str = '',  # Proxy
            chars_strip: str = '',  # Chars to be removed from the edges of DeepSeek responses
            system_prompt: str = '',  # Role
            system_prompt_adv: str = '',  # Additional options (example, JSON-format)
            model: str = '',  # model
            timeout: float = 60,  # timeout
            max_retries: int = 0,  # default is 2
            temperature: float = 1.0,  # default is 1.0
            top_p: float = 1.0,  # default is 1.0
            is_stream: bool = False,
    ) -> None:
        self._chars_strip = chars_strip
        self._client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(proxies={"http://": proxy, "https://": proxy} if proxy else None),
            max_retries=max_retries,
            timeout=timeout,
            base_url=DEEPSEEK_BASE_URL
        )
        self._messages = [
            {
                'role': "system",
                'content': system_prompt + '\n' + system_prompt_adv
            },
        ]
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._is_stream = is_stream

    def get_answer(self, message: str, **replace_texts) -> str:
        """Get text response from DeepSeek by prompt"""

        # Replacing all special keywords to text in message
        for replace_keyword, replace_text in replace_texts.items():
            message = message.replace(replace_keyword, replace_text)

        # Add user message
        self._messages.append({"role": "user", "content": message})

        # Get response from DeepSeek
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                stream=self._is_stream,
                temperature=self._temperature,
                top_p=self._top_p,
            )
        except APITimeoutError:
            yield f"[Error!!! DeepSeek didn't give response with {self._client.timeout:.2f} seconds!]"
            return
        except Exception as e:
            yield f"[Error!!! DeepSeek something wrong: {str(e)}]"
            return

        if self._is_stream:
            text = ''
            for token in response:
                if not token.choices[0].finish_reason:
                    yield str(token.choices[0].delta.content)
                    text += str(token.choices[0].delta.content)
        else:
            text = response.choices[0].message.content.strip(self._chars_strip)
            yield text

        # Remember DeepSeek response
        self._messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    chat_deepseek = DeepSeek(
        api_key=DEEPSEEK_API_KEY,
        model=DEEPSEEK_MODELS["deepseek"],
        chars_strip=RESPONSE_STRIP_CHARS,
        is_stream=True,
    )
    print("Starting dialog with DeepSeek:\n")

    step = 1
    while True:
        print(f"{step:2}. You:", end="\n    ")
        question = input("What do you want to ask DeepSeek: ")
        if not question:
            break

        answer = chat_deepseek.get_answer(question)
        print(f"{step:2}. DeepSeek:", end="\n    ")
        for chunk in answer:
            print(chunk, end="")
        print()

        print('_' * 100, end='\n\n')
        step += 1
