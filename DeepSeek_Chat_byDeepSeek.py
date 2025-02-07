# Позволяет вести диалог с DeepSeek

from deepseek import DeepSeekAPI

DEEPSEEK_API_KEY = 'INSERT_YOUR_API_KEY_FROM_DeepSeek_ACCOUNT'
RESPONSE_STRIP_CHARS = '«»„““”"❝❞„⹂〝〞〟＂‹›❮❯‚‘‘‛’❛❜❟`\'., '


class DeepSeek:
    """DeepSeek chat assistant"""

    def __init__(
            self,
            api_key: str = '',  # DeepSeek Api-key
            chars_strip: str = '',  # Chars to be removed from the edges of DeepSeek responses
            system_prompt: str = '',  # Role
            system_prompt_adv: str = '',  # Additional options (example, JSON-format)
            is_stream: bool = False,
    ) -> None:
        self._chars_strip = chars_strip
        self._client = DeepSeekAPI(api_key=api_key)
        self._messages = [
            {
                'role': "system",
                'content': system_prompt + '\n' + system_prompt_adv
            },
        ]
        self._is_stream = is_stream
        print("Your balance:", self._client.user_balance())

    def get_answer(self, message: str, **replace_texts) -> str:
        """Get text response from DeepSeek by prompt"""

        # Replacing all special keywords to text in message
        for replace_keyword, replace_text in replace_texts.items():
            message = message.replace(replace_keyword, replace_text)

        # Add user message
        self._messages.append({"role": "user", "content": message})

        # Get response from DeepSeek
        try:
            response = self._client.chat_completion(
                prompt=self._messages,
                stream=self._is_stream,
            )
        except Exception as e:
            yield f"[Error!!! DeepSeek something wrong: {str(e)}]"
            return

        if self._is_stream:
            text = ''
            for token in response:
                yield str(token)
                text += str(token)
        else:
            text = response.strip(self._chars_strip)
            yield text

        # Remember DeepSeek response
        self._messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    chat_deepseek = DeepSeek(
        api_key=DEEPSEEK_API_KEY,
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
