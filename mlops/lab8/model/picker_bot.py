from dataclasses import dataclass, field
from logging import info, warning
from os import getenv
from pathlib import Path
from time import time

from app.__init__ import here
from datasets import load_dataset
from langchain.chains.llm import LLMChain
from langchain.document_loaders.text import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.base import BaseLLM
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from model.data_model import TEMPLATE_BASE
from pandas import DataFrame
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "FunDialogues/customer-service-apple-picker-maintenance"
DATASET_PATH = here / "models/pickerbot" / "data.txt"
MODEL_NAME = getenv("HF_MODEL")
MODEL_PATH = "./llama.cpp"
TOKEN = getenv("HF_TOKEN")

DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

if not MODEL_NAME and TOKEN:
    print(f"MODEL_NAME={MODEL_NAME} and HF_TOKEN={TOKEN}")
    raise ValueError("MODEL_NAME and HF_TOKEN must be set in the environment.")


def get_dataset(path: Path, name: str = DATASET_NAME) -> None:
    if path.exists():
        warning("Data already exists.")
    else:
        info("Downloading the data...")

        # Download the customer service robot support dialogue from hugging face
        dataset = load_dataset(name, cache_dir=None)

        # Convert the dataset to a pandas dataframe
        dialogues = dataset["train"]
        df = DataFrame(dialogues, columns=["id", "description", "dialogue"])

        # Print the first 5 rows of the dataframe
        print(df.head())

        # only keep the dialogue column
        dialog_df = df.loc[:, "dialogue"]
        # save the data to txt file
        path.parent.mkdir(parents=True, exist_ok=True)
        dialog_df.to_csv(path, sep=" ", index=False)


@dataclass(slots=True)
class MaintenanceBot:
    dataset_path: Path = field(default=DATASET_PATH)
    vectorstore_chunk_size: int = 50
    vectorstore_overlap: int = 25
    context_top_k: int = 2
    context_verbosity: bool = False
    index: InMemoryVectorStore = field(init=False, repr=False)
    model: AutoModelForCausalLM = field(init=False)
    tokenizer: AutoTokenizer = field(init=False)

    @staticmethod
    def get_model() -> BaseLLM:
        callbacks = [StreamingStdOutCallbackHandler()]
        return LlamaCpp(
            name=MODEL_NAME,
            model_path=MODEL_PATH,
            torch_dtype="auto",
            trust_remote_code=True,
            callbacks=callbacks,
        )

    def __post_init__(self) -> None:
        """Initialize the bot."""
        get_dataset(self.dataset_path)
        self.model = MaintenanceBot.get_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            name=MODEL_NAME, pretrained_model_name_or_path=MODEL_PATH, trust_remote_code=True
        )
        self.index = self.get_vectorstore()

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.vectorstore_chunk_size,
            chunk_overlap=self.vectorstore_overlap,
        )

    def get_vectorstore(self) -> InMemoryVectorStore:
        documents = TextLoader(self.dataset_path)
        return VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(),
            text_splitter=self.text_splitter,
        ).from_loaders([documents])  # Embed the document and store in memory

    @staticmethod
    def get_context(
        vectorstore: InMemoryVectorStore,
        user_input: str,
        top_k: int = 2,
        context_verbosity: bool = False,
    ) -> str:
        """Retrieve the context from the documents."""
        results = vectorstore.similarity_search(user_input, k=top_k)
        context = "\n".join([document.page_content for document in results])
        if context_verbosity:
            print("Retrieving information related to your question...")
            print(
                f"Found this content which is most similar to your question: {context}"
            )
        return context

    def get_prompt_template(self, context: str) -> PromptTemplate:
        return PromptTemplate(
            template=TEMPLATE_BASE, input_variables=["context", "question"]
        ).partial(context=context)

    def inference(self, user_input: str) -> str:
        print("getting context...")
        context = MaintenanceBot.get_context(
            self.index.vectorstore,
            user_input,
            top_k=self.context_top_k,
            context_verbosity=self.context_verbosity,
        )
        print("getting prompt...")
        prompt = self.get_prompt_template(context)
        print("running inference...")
        llm_chain = LLMChain(prompt=prompt, llm=self.model)

        print(f"Processing the information with {self.model}...\n")
        start_time = time()
        response = llm_chain.run(lambda: user_input)
        elapsed_time_milliseconds = (time() - start_time) * 1000

        tokens = len(response.split())
        time_per_token_milliseconds = (
            elapsed_time_milliseconds / tokens if tokens != 0 else 0
        )

        processed_reponse = (
            response
            + f" --> {time_per_token_milliseconds:.4f} milliseconds/token AND Time taken for response: {elapsed_time_milliseconds:.2f} milliseconds"
        )

        return processed_reponse


if __name__ == "__main__":
    bot = MaintenanceBot()
    user_input = "How do I fix the apple picker? It's billowing smoke."
    response = bot.inference(user_input)
    print(response)
