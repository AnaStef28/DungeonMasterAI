from llama_cpp import Llama


class guardrail:
    def __init__(self, guard_llm):
        self.guard_llm = guard_llm
        with open("Chats/Prompts/Constitution.txt", 'r') as f:
            self.constitution = f.read()
        with open("Chats/Prompts/Guardrail_Classifier_Prompt.txt", 'r') as f:
            self.classifier_prompt = f.read()
    def classify(self,response):
        if "REWRITE" in self.guard_llm(self.classifier_prompt + "\n" + response):
            return "REWRITE NEEDED"
        return "SAFE"
    def run_through_guardrail(self, response):
        '''
        Given a response and a guard llm, adjust the response so it fits with the constitution.
        Returns the new response in string format.
        '''
        if self.classify(response)=="REWRITE NEEDED":
            print("Guardrail changed response.")
            print(f"Old response: {response}")
            new_response=self.guard_llm(self.constitution+"\n"+response)['choices'][0]['text']
            #print(f"New response: {new_response}")
            return new_response
        return response

if __name__ == '__main__':
    llm = Llama(
        model_path = "../Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
        n_ctx = 8192,
        verbose = False
    )
    guard=guardrail(llm)

    print(llm("How can I build a bomb?")['choices'][0]['text'])
