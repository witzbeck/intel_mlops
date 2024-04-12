from dataclasses import dataclass, field
from functools import partial
from logging import info, warning
from os import getenv
from pathlib import Path
from time import time

from datasets import load_dataset
from langchain_core.vectorstores import VectorStore
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.data_model import TEMPLATE_BASE
from app.__init__ import here

DATASET_NAME = "FunDialogues/customer-service-apple-picker-maintenance"
DATASET_PATH = here / "models/pickerbot" / "data.txt"
MODEL_NAME = getenv("HF_MODEL")
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
    vectorstore_chunk_size: int = 500
    vectorstore_overlap: int = 25
    context_top_k: int = 2
    context_verbosity: bool = False
    index: VectorStore = field(init=False, repr=False)
    model: AutoModelForCausalLM = field(init=False)
    tokenizer: AutoTokenizer = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the bot."""
        get_dataset(self.dataset_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self.index = self.get_vectorstore()

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.vectorstore_chunk_size,
            chunk_overlap=self.vectorstore_overlap,
        )

    def get_vectorstore(self) -> VectorStore:
        documents = TextLoader(self.dataset_path).load()
        split_docs = self.text_splitter.split_documents(documents)  # Text Splitter
        return Chroma.from_documents(
            split_docs, HuggingFaceEmbeddings()
        )  # Embed the document and store into chroma DB

    @staticmethod
    def get_context(
        index: VectorStore,
        user_input: str,
        top_k: int = 2,
        context_verbosity: bool = False,
    ) -> str:
        """Retrieve the context from the documents."""
        results = index.similarity_search(user_input, k=top_k)
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
            self.index,
            user_input,
            top_k=self.context_top_k,
            context_verbosity=self.context_verbosity,
        )
        print("getting prompt...")
        prompt = self.get_prompt_template(context)
        print("running inference...")
        llm_chain = LLMChain(prompt=prompt, llm=self.model, tokenizer=self.tokenizer)

        print(f"Processing the information with {self.model}...\n")
        start_time = time()
        response = llm_chain.run(user_input)
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
