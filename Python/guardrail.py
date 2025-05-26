class guardrail:
    def __init__(self):
        with open("Chats/Prompts/Constitution.txt", 'r') as f:
            self.constitution = f.read()
        with open("Chats/Prompts/Guardrail_Classifier_Prompt.txt", 'r') as f:
            self.classifier_prompt = f.read()

    def run_through_guardrail(self, response, guard_llm):
        '''
        Given a response and a guard llm, adjust the response so it fits with the constitution.
        Returns the new response in string format.
        '''
        if guard_llm(self.classifier_prompt+"\n"+response)=="REWRITE NEEDED":
            #print("Guardrail changed response.")
            #print(f"Old response: {response}")
            new_response=guard_llm(self.constitution+"\n"+response)['choices'][0]['text']
            #print(f"New response: {new_response}")
            return new_response
        return response