from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from src.language_model.gemini.gemini_llm import GeminiLLM
from src.language_model.openai.openai_llm import OpenAILLM
from src.language_model.huggingface.huggingface_endpoint.huggingface_endpoint_llm import HuggingFaceEndpointLLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMChatbot:

    def __init__(self, llm_provider):
        self.llm_provider= llm_provider
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chat_template= ChatPromptTemplate([
                                                ("system", "You are a helpful assistant"),
                                                MessagesPlaceholder(variable_name= "chat_history"),
                                                ("human", "{query}")
                                                ])
        self.model= self._get_llm_instance()

    def _get_llm_instance(self):
        """
        Initializes the appropriate LLM provider
        """
        providers={
                    "google-gemini":GeminiLLM,
                    "open-ai": OpenAILLM,
                    "huggingface-endpoint": HuggingFaceEndpointLLM,
                  }
        if self.llm_provider not in providers:
            raise ValueError(f"Invalid LLM provider passed. Please select one of these : {', '.join(providers.keys())}")

        return providers[self.llm_provider]()
    
    def invoke(self, prompt:str):
        """
        Invokes the loaded model with a prompt
        """
        # Store user input in memory
        self.memory.chat_memory.add_user_message(prompt)
        # Generate formatted prompt using chat history from memory
        formatted_prompt = self.chat_template.invoke({
            "chat_history": self.memory.load_memory_variables({})["chat_history"],
            "query": prompt
        })
        response = self.model.invoke(formatted_prompt)
        # Store model response in memory
        self.memory.chat_memory.add_ai_message(response)
        return response
        # , self.memory.chat_memory


# Example usage
if __name__ == "__main__":
    llm = LLMChatbot("google-gemini")
    while True:
        user_input = input("Enter you text: ")
        if user_input == 'exit':
            break
        result, chat_memory = llm.invoke(user_input)
        print("\n ******", chat_memory)
        print(type(chat_memory.messages[0]), '\n')
        print(result)
