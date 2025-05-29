from llama_cpp import Llama

from dm import *

if __name__ == '__main__':
    llm = Llama(
        model_path = "../Qwen3-8B-Q3_K_L.gguf",
        n_ctx = 8192,
        verbose=False
    )
    current_dm=dm(llm,llm)
    current_dm.start()

