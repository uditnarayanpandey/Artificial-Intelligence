from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Abstract class for LLM models
    """

    @abstractmethod
    def load_model(self):
        """
        Loads the specific LLM model
        """
        pass

    @abstractmethod
    def invoke(self, prompt:str):
        """
        Process a given prompt and return the model's response
        """
        pass