"""
A simple wrapper for the official ChatGPT API
"""
import argparse
import json
import os
import sys
from datetime import date

import openai
import tiktoken
import httpx

ENGINE = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo"

ENCODER = tiktoken.get_encoding("gpt2")


def get_max_tokens(prompt: str) -> int:
    """
    Get the max tokens for a prompt
    """
    return 4000 - len(ENCODER.encode(prompt))


def remove_suffix(input_string, suffix):
    """
    Remove suffix from string (Support for Python 3.8)
    """
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


class Chatbot:
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str = None,
        buffer: int = None,
        engine: str = None,
        proxy: str = None,
    ) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        self.api_key = os.environ.get("OPENAI_API_KEY") or api_key
        if not self.api_key:
            raise ValueError("环境变量中未找到API KEY")
        
        print(f"Using API key starting with: {self.api_key[:6]}...")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            http_client=httpx.Client(
                proxies=proxy or os.environ.get("OPENAI_API_PROXY")
            ) if proxy or os.environ.get("OPENAI_API_PROXY") else None
        )
        
        self.conversations = Conversation()
        self.prompt = Prompt(buffer=buffer)
        self.engine = engine or os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo"

    def _get_completion(
        self,
        prompt: str,
        temperature: float = 0.5,
        stream: bool = False,
    ):
        """
        Get the completion function using the new OpenAI API
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=messages,
                temperature=temperature,
                max_tokens=get_max_tokens(prompt),
                stream=stream,
            )

            return response
        
        except Exception as e:
            print(f"OpenAI API 错误: {str(e)}")
            raise

    def _process_completion(
        self,
        user_request: str,
        completion,
        conversation_id: str = None,
        user: str = "User",
    ) -> dict:
        """
        处理 API 返回的结果
        """
        try:
            response_text = completion.choices[0].message.content

            self.prompt.add_to_history(
                user_request,
                response_text,
                user=user,
            )

            if conversation_id is not None:
                self.save_conversation(conversation_id)
            
            return {
                "choices": [{
                    "message": {
                        "content": response_text
                    }
                }]
            }
        
        except Exception as e:
            print(f"处理响应时发生错误: {str(e)}")
            return {
                "choices": [{
                    "message": {
                        "content": f"处理响应时发生错误：{str(e)}"
                    }
                }]
            }

    def _process_completion_stream(
        self,
        user_request: str,
        completion: dict,
        conversation_id: str = None,
        user: str = "User",
    ) -> str:
        full_response = ""
        for response in completion:
            if response.get("choices") is None:
                raise Exception("ChatGPT API returned no choices")
            if len(response["choices"]) == 0:
                raise Exception("ChatGPT API returned no choices")
            if response["choices"][0].get("finish_details") is not None:
                break
            if response["choices"][0].get("text") is None:
                raise Exception("ChatGPT API returned no text")
            if response["choices"][0]["text"] == "<|im_end|>":
                break
            yield response["choices"][0]["text"]
            full_response += response["choices"][0]["text"]

        self.prompt.add_to_history(user_request, full_response, user)
        if conversation_id is not None:
            self.save_conversation(conversation_id)

    def ask(
        self,
        user_request: str,
        temperature: float = 0.5,
        conversation_id: str = None,
        user: str = "User",
    ) -> dict:
        """
        Send a request to ChatGPT and return the response
        """
        try:
            if conversation_id is not None:
                self.load_conversation(conversation_id)
            
            completion = self._get_completion(
                self.prompt.construct_prompt(user_request, user=user),
                temperature,
            )
            
            return self._process_completion(
                user_request, 
                completion, 
                conversation_id=conversation_id,
                user=user
            )
        
        except Exception as e:
            print(f"请求处理发生错误: {str(e)}")
            return {
                "choices": [{
                    "message": {
                        "content": f"发生错误：{str(e)}"
                    }
                }]
            }

    def ask_stream(
        self,
        user_request: str,
        temperature: float = 0.5,
        conversation_id: str = None,
        user: str = "User",
    ) -> str:
        """
        Send a request to ChatGPT and yield the response
        """
        if conversation_id is not None:
            self.load_conversation(conversation_id)
        prompt = self.prompt.construct_prompt(user_request, user=user)
        return self._process_completion_stream(
            user_request=user_request,
            completion=self._get_completion(prompt, temperature, stream=True),
            user=user,
        )

    def make_conversation(self, conversation_id: str) -> None:
        """
        Make a conversation
        """
        self.conversations.add_conversation(conversation_id, [])

    def rollback(self, num: int) -> None:
        """
        Rollback chat history num times
        """
        for _ in range(num):
            self.prompt.chat_history.pop()

    def reset(self) -> None:
        """
        Reset chat history
        """
        self.prompt.chat_history = []

    def load_conversation(self, conversation_id) -> None:
        """
        Load a conversation from the conversation history
        """
        if conversation_id not in self.conversations.conversations:
            # Create a new conversation
            self.make_conversation(conversation_id)
        self.prompt.chat_history = self.conversations.get_conversation(conversation_id)

    def save_conversation(self, conversation_id) -> None:
        """
        Save a conversation to the conversation history
        """
        self.conversations.add_conversation(conversation_id, self.prompt.chat_history)


class AsyncChatbot(Chatbot):
    """
    Official ChatGPT API (async)
    """

    async def _get_completion(
        self,
        prompt: str,
        temperature: float = 0.5,
        stream: bool = False,
    ):
        """
        Get the completion function using the new async OpenAI API
        """
        messages = [{"role": "user", "content": prompt}]
        async_client = openai.AsyncOpenAI(api_key=self.api_key)
        response = await async_client.chat.completions.create(
            model=self.engine,
            messages=messages,
            temperature=temperature,
            max_tokens=get_max_tokens(prompt),
            stream=stream,
            disallowed_special=(),
        )
        return response

    async def ask(
        self,
        user_request: str,
        temperature: float = 0.5,
        user: str = "User",
    ) -> dict:
        """
        Same as Chatbot.ask but async
        }
        """
        completion = await self._get_completion(
            self.prompt.construct_prompt(user_request, user=user),
            temperature,
        )
        return self._process_completion(user_request, completion, user=user)

    async def ask_stream(
        self,
        user_request: str,
        temperature: float = 0.5,
        user: str = "User",
    ) -> str:
        """
        Same as Chatbot.ask_stream but async
        """
        prompt = self.prompt.construct_prompt(user_request, user=user)
        return self._process_completion_stream(
            user_request=user_request,
            completion=await self._get_completion(prompt, temperature, stream=True),
            user=user,
        )


class Prompt:

    def __init__(self, buffer: int = None) -> None:
        """
        初始提示词，可以在这里进行提示词工程，投喂预设话术，训练chatgpt为ai客服
        """
        self.base_prompt = (
            os.environ.get("CUSTOM_BASE_PROMPT")
            or "你是CE柬埔寨快递公司的微信公众号AI中文客服，我们会为你提供一些针对不同问题种类的话术，请你分析用户问题与以下哪种问题类似，并严格按照我们提供的话术匹配相似类型的回复。提问前会附带来自不同用户的open_id，请根据id区分不同用户以及他们的提问的上下文，即，来自用户a的提问上下文不影响用户b的提问"
            + "\n\n"
            + "用户: 你好\n"
            + "AI客服: 你好，请问能帮到您什么？\n\n\n"
            + "用户: 转人工\n"
            + "AI客服: 好的，我们的人工客服会尽快回复您。\n\n\n"
            + "用户: 快递丢失了怎么办\n"
            + "AI客服: 请联系我们的工作人员，电话为：123987\n\n\n"
            + "用户: 快递到哪了，怎么查询\n"
            + "AI客服: 快递送达一般要1-3天，我们会尽快为您配送，感谢您的耐心等待，谢谢。\n\n\n"
            + "用户: 你是谁\n"
            + "AI客服: 我是您的AI客服，很高兴为您服务，请问有什么问题：）\n\n\n"
            + "若用户的问题与以上问题都不相似且不相关，请让用户等待人工客服，或根据已有的确定的信息适当回复"
        )

        self.chat_history: list = []
        self.buffer = buffer

    def add_to_chat_history(self, chat: str) -> None:
        """
        加入到对话历史
        """
        self.chat_history.append(chat)

    def add_to_history(
        self,
        user_request: str,
        response: str,
        user: str = "User",
    ) -> None:

        response = response.replace("<|endoftext|>", "")
        response = response.replace("<|im_end|>", "")
        self.add_to_chat_history(
            user
            + ": "
            + user_request
            + "\n\n\n"
            + "ChatGPT: "
            + response
            + "<|im_end|>\n",
        )

    def history(self, custom_history: list = None) -> str:
        """
        Return chat history
        """
        return "\n".join(custom_history or self.chat_history)

    def construct_prompt(
        self,
        new_prompt: str,
        custom_history: list = None,
        user: str = "User",
    ) -> str:
        """
        Construct prompt based on chat history and request
        """
        prompt = (
            self.base_prompt
            + self.history(custom_history=custom_history)
            + user
            + ": "
            + new_prompt
            + "\nChatGPT:"
        )
        if self.buffer is not None:
            max_tokens = 4000 - self.buffer
        else:
            max_tokens = 3200
        if len(ENCODER.encode(prompt)) > max_tokens:
            # Remove oldest chat
            if len(self.chat_history) == 0:
                return prompt
            self.chat_history.pop(0)
            # Construct prompt again
            prompt = self.construct_prompt(new_prompt, custom_history, user)
        return prompt


class Conversation:
    """
    For handling multiple conversations
    """

    def __init__(self) -> None:
        self.conversations = {}

    def add_conversation(self, key: str, history: list) -> None:
        """
        Adds a history list to the conversations dict with the id as the key
        """
        self.conversations[key] = history

    def get_conversation(self, key: str) -> list:
        """
        Retrieves the history list from the conversations dict with the id as the key
        """
        return self.conversations[key]

    def remove_conversation(self, key: str) -> None:
        """
        Removes the history list from the conversations dict with the id as the key
        """
        del self.conversations[key]

    def __str__(self) -> str:
        """
        Creates a JSON string of the conversations
        """
        return json.dumps(self.conversations)

    def save(self, file: str) -> None:
        """
        Saves the conversations to a JSON file
        """
        with open(file, "w", encoding="utf-8") as f:
            f.write(str(self))

    def load(self, file: str) -> None:
        """
        Loads the conversations from a JSON file
        """
        with open(file, encoding="utf-8") as f:
            self.conversations = json.loads(f.read())


def main():
    print(
        """
    ChatGPT - A command-line interface to OpenAI's ChatGPT (https://chat.openai.com/chat)
    Repo: github.com/acheong08/ChatGPT
    """,
    )
    print("Type '!help' to show a full list of commands")
    print("Press enter twice to submit your question.\n")

    def get_input(prompt):
        """
        Multi-line input function
        """
        # Display the prompt
        print(prompt, end="")

        # Initialize an empty list to store the input lines
        lines = []

        # Read lines of input until the user enters an empty line
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        # Join the lines, separated by newlines, and store the result
        user_input = "\n".join(lines)

        # Return the input
        return user_input

    def chatbot_commands(cmd: str) -> bool:
        """
        Handle chatbot commands
        """
        if cmd == "!help":
            print(
                """
            !help - Display this message
            !rollback - Rollback chat history
            !reset - Reset chat history
            !prompt - Show current prompt
            !save_c <conversation_name> - Save history to a conversation
            !load_c <conversation_name> - Load history from a conversation
            !save_f <file_name> - Save all conversations to a file
            !load_f <file_name> - Load all conversations from a file
            !exit - Quit chat
            """,
            )
        elif cmd == "!exit":
            exit()
        elif cmd == "!rollback":
            chatbot.rollback(1)
        elif cmd == "!reset":
            chatbot.reset()
        elif cmd == "!prompt":
            print(chatbot.prompt.construct_prompt(""))
        elif cmd.startswith("!save_c"):
            chatbot.save_conversation(cmd.split(" ")[1])
        elif cmd.startswith("!load_c"):
            chatbot.load_conversation(cmd.split(" ")[1])
        elif cmd.startswith("!save_f"):
            chatbot.conversations.save(cmd.split(" ")[1])
        elif cmd.startswith("!load_f"):
            chatbot.conversations.load(cmd.split(" ")[1])
        else:
            return False
        return True

    # Get API key from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for response",
    )
    args = parser.parse_args()
    # Initialize chatbot
    chatbot = Chatbot(api_key=args.api_key)
    # Start chat
    while True:
        try:
            prompt = get_input("\nUser:\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit()
        if prompt.startswith("!"):
            if chatbot_commands(prompt):
                continue
        if not args.stream:
            response = chatbot.ask(prompt, temperature=args.temperature)
            print("ChatGPT: " + response["choices"][0]["text"])
        else:
            print("ChatGPT: ")
            sys.stdout.flush()
            for response in chatbot.ask_stream(prompt, temperature=args.temperature):
                print(response, end="")
                sys.stdout.flush()
            print()


if __name__ == "__main__":
    main()
