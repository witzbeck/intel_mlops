from collections.abc import Generator
from dataclasses import dataclass, field
from functools import cached_property
from logging import info, warning
from os import getenv
from pathlib import Path
from threading import Thread
from time import time

from datasets import load_dataset
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.document_loaders.text import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from pandas import DataFrame
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)

from app.__init__ import here
from model.data_model import TEMPLATE_BASE

DATASET_NAME = "FunDialogues/customer-service-apple-picker-maintenance"
DATASET_PATH = here / "models/pickerbot" / "data.txt"
MAX_NEW_TOKENS = 512
MODEL_NAME = getenv("HF_MODEL")
TOKEN = getenv("HF_TOKEN")

DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

if not MODEL_NAME and TOKEN:
    print(f"MODEL_NAME={MODEL_NAME} and HF_TOKEN={TOKEN}")
    raise ValueError("MODEL_NAME and HF_TOKEN must be set in the environment.")


def get_dataset(
    path: Path = DATASET_PATH, name: str = DATASET_NAME, show_rows: int = 5
) -> None:
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
        print(df.head(show_rows))

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
    index: VectorStoreIndexWrapper = field(init=False, repr=False)
    vectorstore: InMemoryVectorStore = field(init=False, repr=False)
    model: AutoModelForCausalLM = field(init=False)
    tokenizer: AutoTokenizer = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the bot."""
        if not DATASET_PATH.exists():
            get_dataset()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, padding_side="left"
        )
        self.index = self.get_vectorstore_index()
        self.vectorstore = self.index.vectorstore

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.vectorstore_chunk_size,
            chunk_overlap=self.vectorstore_overlap,
        )

    def get_vectorstore_index(self) -> VectorStoreIndexWrapper:
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
        return PromptTemplate.from_template(template=TEMPLATE_BASE).partial(
            context=context
        )

    def inference(self, question: str) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=300.0,
        )
        print("getting context...")
        context = MaintenanceBot.get_context(
            self.vectorstore,
            user_input=question,
            top_k=self.context_top_k,
            context_verbosity=self.context_verbosity,
        )
        print("getting prompt...")
        prompt = self.get_prompt_template(context)

        print("creating pipeline...")
        rag_pipeline = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                device_map="cpu",
                max_new_tokens=MAX_NEW_TOKENS,
            )
        )

        print("running inference...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=rag_pipeline,
            retriever=VectorStoreRetriever(
                vectorstore=self.vectorstore, search_kwargs={"top_k": 2}
            ),
            # chain_type_kwargs={"prompt": prompt},
        )
        print(f"Processing the information with {self.model}...\n")
        start_time = time()
        qa_chain.invoke({"query": f"{question}"})

        response = ""
        tokens = 0
        for token in streamer:
            response += token
            tokens += 1
            elapsed_time_ms = (time() - start_time) * 1000

        time_per_token_ms = elapsed_time_ms / tokens if tokens != 0 else 0

        processed_reponse = (
            response
            + f" --> {time_per_token_ms:.4f} ms/token \nTime taken for response: {elapsed_time_ms:.2f} ms"
        )

        return processed_reponse


if __name__ == "__main__":
    bot = MaintenanceBot(context_verbosity=True)
    user_input = "How do I fix the apple picker? It's billowing smoke."
    response = bot.inference(user_input)
    print(response)
