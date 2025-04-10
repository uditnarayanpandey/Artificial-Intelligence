import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from src.language_model.base.base_llm import BaseLLM

class HuggingFaceEndpointLLM:
    def __init__(self):
        try:
            self.api_key= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        except KeyError:
            raise EnvironmentError("HUGGINGFACEHUB_ACCESS_TOKEN not found as an env variable")
        else:
            llm = HuggingFaceEndpoint(
                                        repo_id="meta-llama/Llama-3.2-3B-Instruct",
                                        task="text-generation",
                                        )
            self.model= ChatHuggingFace(llm=llm)

    def load_model(self):
        """
        Returns the loaded model
        """
        return self.model

    def invoke(self, prompt:str):
        """
        Sends a prompt to HuggingFace Model and returns the response
        """
        return self.model.invoke(prompt).content