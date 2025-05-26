from guardrail import guardrail
from llama_cpp import Llama
from pathlib import Path
from dataFunctions import *


def get_context_prompt(question):
    '''
    Get a prompt + context.
    Returns the full string.
    '''
    embedder, index, metadata = prepare_embeddings()
    context = retrieve_context(question, embedder, index, metadata)
    return build_prompt([context],question)


def continue_chat(chat_file=None):
    if chat_file is None:
        #this is a new chat
        user_query = {
            "role": "user",
            "content": get_context_prompt(input("Query:"))
        }
        title = llm("Write a title based on this question: " + user_query['content'])['choices'][0]['text'] + ".json"
        with (open('Chats/Prompts/Initial_Prompt.txt', 'r') as file):
            initial = {
                "role": "system",
                "content": file.read()
            }
        chat_history = [initial, user_query]
        chat_file=title
    else:
        with open(chat_file, 'r') as file:
            chat_history = json.load(file)
    query="b"
    while len(query) > 0:
        response = llm.create_chat_completion(messages = chat_history)
        new_response = guard.run_through_guardrail(response['choices'][0]['message']['content'], llm)
        print(new_response)
        chat_history.append({
                "role": "assistant",
                "content": new_response
            })
        with open(chat_file, 'w', encoding = 'utf-8') as file:
            json.dump(chat_history, file, ensure_ascii = False, indent = 2)
        user_query = {
            "role": "user",
            "content": get_context_prompt(input("Query:"))
        }
        chat_history.append(user_query)
        #I am a mage, my friend is a rogue. How does out adventure start?


if __name__ == '__main__':
    # Base path relative to this script's location
    base_path = Path(__file__).resolve().parent

    # Path to model file (relative to script)
    model_path = base_path.parent / 'Hermes-3-Llama-3.2-3B.Q4_K_M.gguf'

    prompt_path = base_path / 'Chats' / 'Prompts' / 'Constitution.txt'

    llm = Llama(
        model_path=str(model_path),
        n_ctx=8192,
        verbose=False
    )

    guard = guardrail(str(prompt_path))

    continue_chat()



