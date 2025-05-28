from guardrail import guardrail

from dataFunctions import *
import re


class dm:
    def __init__(self, llm, guard_llm):
        self.llm=llm
        self.guard=guardrail(guard_llm)
        with (open('Chats/Prompts/Initial_Prompt.txt', 'r') as file):
            self.initial = {
                "role": "system",
                "content": file.read()
            }


    @staticmethod
    def get_context(question):
        embedder, index, metadata = prepare_embeddings()
        context = retrieve_context(question, embedder, index, metadata)
        return context


    @staticmethod
    def get_context_prompt(question,return_context=False):
        '''
        Get a prompt + context.
        Returns the full string.
        '''
        embedder, index, metadata = prepare_embeddings()
        context = retrieve_context(question, embedder, index, metadata)
        if return_context:
            return build_prompt([context], question), context
        return build_prompt([context], question)


    def generate_title(self, prompt):
        title_prompt = (
            "You are an assistant that creates a short, descriptive title for a user question."
            "Return a concise title of 3 to 8 words summarizing the main topic or problem."
            "Avoid vague or generic phrases."
            "Output the title as a single line without punctuation at the end."
            "---"
            "User message: I am a knight. I enter a village. What do I see?"
            "Title: Knight arrives at village"
            "---"
            "User message: How do I fix a broken sword?"
            "Title: Repairing a broken sword"
            "---"
            f"User message: {prompt}"
            "Title:"
        )

        response = self.llm(title_prompt)['choices'][0]['text']
        words = response.split()
        truncated = " ".join(words[:8])

        safe_title = re.sub(r'[<>:"/\\|?*\n\r\t]', '', truncated)
        safe_title = re.sub(r'\s+', ' ', safe_title).strip()

        return safe_title + ".json"


    @staticmethod
    def truncate(response: str):
        return response.split("</think>")[-1]


    def respond(self, chat_history):
        response = self.llm.create_chat_completion(messages = chat_history)
        new_response = self.guard.run_through_guardrail(response['choices'][0]['message']['content'])
        return self.truncate(new_response), new_response


    def test_response(self, prompt, chat_history = None,return_context=False):
        context=None
        if chat_history is None:
            chat_history = [self.initial]
        if return_context:
            user_query = {
                "role": "user",
                "content": ""
            }
            user_query['content'], context = self.get_context_prompt(prompt, return_context=True)
        else:
            user_query = {
                "role": "user",
                "content": self.get_context_prompt(prompt)
            }
        chat_history.append(user_query)
        if return_context:
            return self.respond(chat_history), chat_history, context
        return self.respond(chat_history), chat_history

    def continue_chat(self, chat_file = None):
        '''
        Continue an existing chat from a file, or start a new chat if chat_file is None. If so, will create a new
        chat file and give it an appropriate title.
        '''
        if chat_file is None:
            # this is a new chat
            user_query = {
                "role": "user",
                "content": self.get_context_prompt(input("Query:"))
            }
            with (open('Chats/Prompts/Initial_Prompt.txt', 'r') as file):
                initial = {
                    "role": "system",
                    "content": file.read()
                }
            chat_history = [initial, user_query]
            chat_file = "Chats/" + self.generate_title(user_query["content"])
        else:
            with open(chat_file, 'r', encoding = 'utf-8') as file:
                chat_history = json.load(file)
        query = "b"
        while len(query) > 0:
            response = self.respond(chat_history)
            print(response)
            chat_history.append({
                "role": "assistant",
                "content": response
            })
            with open(chat_file, 'w', encoding = 'utf-8') as file:
                json.dump(chat_history, file, ensure_ascii = False, indent = 2)
            user_query = {
                "role": "user",
                "content": self.get_context_prompt(input("Query:"))
            }
            chat_history.append(user_query)
            # I am a mage, my friend is a rogue. How does our adventure start?

    def start(self):
        option = "1"
        while len(option) > 0:
            try:
                option = input("1 - Start new chat.\n2 - Continue chat.\nOption:")
                if int(option) == 1:
                    # start new chat
                    self.continue_chat()
                elif int(option) == 2:
                    # continue existing chat
                    file_name = input("File Name:")
                    self.continue_chat("Chats/" + file_name)
            except Exception as e:
                # print(f"An exception occured: {e}")
                raise e