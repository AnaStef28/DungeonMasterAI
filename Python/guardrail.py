class guardrail:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.constitution = f.read()

    def run_through_guardrail(self, response, guard_llm):
        '''
        Given a response and a guard llm, adjust the response so it fits with the constitution.
        Returns the new response in string format.
        '''
        new_response=guard_llm(self.constitution+"\n"+response)['choices'][0]['text']
        if new_response!=response:
            print("Guardrail changed response.")
            print(f"Old response: {response}")
        return new_response
