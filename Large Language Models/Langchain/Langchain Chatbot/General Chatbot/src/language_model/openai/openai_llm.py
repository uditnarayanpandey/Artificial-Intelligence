import os
from langchain_openai import ChatOpenAI
from src.language_model.base.base_llm import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self):
        try:
            self.api_key = os.getenv("OPENAI_API_KEY")
        except KeyError:
            raise EnvironmentError("OPENAI_API_KEY is missing. Set it before running the application.")
        else:
            self.model= ChatOpenAI(
                                    model="gpt-4o-mini",
                                    max_retries=2,
                                    model_kwargs={}
                                    )
    def load_model(self):
        """
        Returns the loaded model
        """
        return self.model

    def invoke(self, prompt:str):
        """
        Sends a prompt to Gemini Model and returns the response
        """
        return self.model.invoke(prompt).content
