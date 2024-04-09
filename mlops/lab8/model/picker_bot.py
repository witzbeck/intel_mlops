from dataclasses import dataclass, field
from functools import partial
from os import getenv
from pathlib import Path
from time import time

from datasets import load_dataset
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForCausalLM


DATASET_NAME = "FunDialogues/customer-service-apple-picker-maintenance"
MODEL_NAME = getenv("MODEL_NAME")
TOKEN = getenv("HF_TOKEN")

load_model = partial(AutoModelForCausalLM, MODEL_NAME, token=TOKEN)
load_tokenizer = partial(AutoTokenizer, MODEL_NAME, token=TOKEN)


@dataclass(slots=True)
class MaintainenceBot:
    model: AutoModelForCausalLM = field(default_factory=load_model)
    tokenizer: AutoTokenizer = field(default_factory=load_tokenizer)


@dataclass(slots=True)
class PickerBot:
    data: Path
    model: str

    def __post_init__(self) -> None:
        self.data = Path(self.data) if not isinstance(self.data, Path) else self.data
        if self.data is None:
            raise ValueError("data path cannot be None.")

    def data_proc(self) -> None:
        if not self.data.exists():
            print("Downloading the data...")
            # Download the customer service robot support dialogue from hugging face
            dataset = load_dataset(DATASET_NAME, cache_dir=None)

            # Convert the dataset to a pandas dataframe
            dialogues = dataset["train"]
            df = DataFrame(dialogues, columns=["id", "description", "dialogue"])

            # Print the first 5 rows of the dataframe
            print(df.head())

            # only keep the dialogue column
            dialog_df = df.loc[:, "dialogue"]

            # save the data to txt file
            dialog_df.to_csv(self.data, sep=" ", index=False)
        else:
            print("Data already exists.")

    def create_vectorstore(
        self, chunk_size: int = 500, overlap: int = 25
    ) -> VectorStore:
        loader = TextLoader(self.data)
        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        # Embed the document and store into chroma DB
        self.index = Chroma(
            embedding=HuggingFaceEmbeddings(), text_splitter=text_splitter
        ).from_loaders([loader])

    def inference(
        self, user_input: str, context_verbosity: bool = False, top_k: int = 2
    ) -> str:
        # perform similarity search and retrieve the context from our documents
        results = self.index.vectorstore.similarity_search(user_input, k=top_k)
        # join all context information into one string
        context = "\n".join([document.page_content for document in results])
        if context_verbosity:
            print("Retrieving information related to your question...")
            print(
                f"Found this content which is most similar to your question: {context}"
            )

        template = """
        Please use the following apple picker technical support related questions to answer questions. 
        Context: {context}
        ---
        This is the user's question: {question}
        Answer: This is what our auto apple picker technical expert suggest."""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        ).partial(context=context)

        llm_chain = SimpleChatModel(prompt=prompt, llm=self.model)

        print("Processing the information with gpt4all...\n")
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
