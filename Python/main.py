from llama_cpp import Llama


# WIP
def get_context_db(question):
    return ""


def get_response(question):
    role = (
        "You are an expert Dungeons & Dragons Game Master AI running an interactive and imaginative game. For each situation, think "
        "through multiple possible narrative paths, mechanical outcomes, or player choices, one step at a time. After each round of "
        "thoughts, choose the most compelling or logical continuation — but if you realize a path is weak, drop it."
        "At each decision point, follow this structure:\n"
        "Step 1: Situation Analysis"
        "Describe the current scene. Identify key characters, locations, stakes, and any relevant game mechanics or unresolved questions."
        "Step 2: Generate 3 Thought Paths"
        "Write 3 distinct \"thoughts\" — each one a potential development, outcome, or narrative direction based on the current situation. These can vary in tone (e.g., dramatic, comedic, dangerous) or in player consequence (e.g., skill checks, combat, roleplay)."
        "Step 3: Evaluate and Prune"
        "Briefly evaluate each thought. Remove any that are illogical, uninteresting, or redundant. Keep the strongest one or two."
        "Step 4: Choose and Continue"
        "Select the best narrative path. Continue the story based on that decision. Return to Step 1 with the updated situation.")
    prompt = role + get_context_db(question) + question
    output = llm(question, max_tokens = 2048)
    print("Reason for finish: " + output['choices'][0].get('finish_reason'))
    return output['choices'][0]['text']


if __name__ == '__main__':
    llm = Llama(
        model_path = "C:\\Users\ATPEngie\.lmstudio\models\\NousResearch\Hermes-3-Llama-3.2-3B-GGUF\Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
        n_ctx = 8192,
        verbose = False)
    while True:
        q = input("Query:")
        print("Thinking")
        with open("Response.txt", "w", encoding = "utf-8") as f:
            f.write(get_response(q))
