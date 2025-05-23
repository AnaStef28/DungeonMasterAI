import json
from llama_cpp import Llama
from dataFunctions import *


def get_context_prompt(question):
    '''
    Get a prompt + context.
    Returns the full string.
    '''
    embedder, index, metadata = prepare_embeddings()
    context = retrieve_context(question, embedder, index, metadata)
    return build_prompt([context],question)


def write(string):
    print(string)
    return True

def start_chat(llm):
    user_prompt=input('Query:')
    query= {
              "role": "user",
              "content": input("Query:")
          }
    title=llm("Write a title based on this question: "+ query['content'])+".json"
    with open('Initial.json', 'r') as file:
        initial = json.load(file)
    chat_history=[initial, query]
    
    response=llm.create_chat_completion(messages = chat_history)
    while not write(response):
        response = llm.create_chat_completion(messages = chat_history)
    chat_history.append(response)
    
    with open(title,'w') as f:
        json.dump(chat_history,f, encoding='utf-8', ensure_ascii=False)
    
    continue_chat(llm,title,chat_history)

def continue_chat(llm, chat_file,chat_history=None):
    if chat_history is None:
        with open(chat_file, 'r') as file:
            chat_history = json.load(file)
    query="b"
    while len(query) > 0:
        query=input("Query:")
        user_query = {
            "role": "user",
            "content": get_context_prompt(input("Query:"))
        }
        chat_history.append(user_query)
        response = llm.create_chat_completion(messages = chat_history)
        print(response['content'])
        chat_history.append(response)
        with open(chat_file, 'a') as file:
            json.dump(chat_history, file, encoding='utf-8', ensure_ascii=False)

if __name__ == '__main__':
    llm = Llama(
        model_path = "../Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
        n_ctx = 8192
    )
    start_chat(llm)



