from random import choice
from unittest import TestCase, main

from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.picker_bot import MaintenanceBot


class TestBot(TestCase):
    """Test the MaintenanceBot class."""

    def setUp(self) -> None:
        """Set up the test."""
        self.prompt_options = [
            "France",
            "Germany",
            "Italy",
            "Spain",
            "United Kingdom",
            "United States",
            "Japan",
            "China",
            "India",
            "Brazil",
            "Russia",
            "Australia",
            "Canada",
            "South Korea",
            "Mexico",
            "Indonesia",
            "Turkey",
            "Netherlands",
            "Saudi Arabia",
            "Switzerland",
            "Sweden",
            "Poland",
            "Belgium",
            "Norway",
            "Austria",
            "Ukraine",
            "Denmark",
            "Finland",
            "Greece",
            "Portugal",
            "Czech Republic",
            "Romania",
            "Hungary",
            "Ireland",
            "New Zealand",
            "Singapore",
            "South Africa",
            "Argentina",
        ]
        self.prompt_input = (f"What is the capital of {choice(self.prompt_options)}?")
        self.bot = MaintenanceBot()

    def test_post_init_steps(self):
        """Test the get_context method."""
        self.assertTrue(self.bot.dataset_path.exists())
        self.assertIsNotNone(self.bot.model)
        self.assertIsNotNone(self.bot.tokenizer)
        self.assertIsNotNone(self.bot.index)
        self.assertIsInstance(self.bot.text_splitter, RecursiveCharacterTextSplitter)
        self.assertIsInstance(self.bot.index, VectorStore)
        self.assertIsInstance(self.bot.model, AutoModelForCausalLM)
        self.assertIsInstance(self.bot.tokenizer, AutoTokenizer)

    def test_get_context(self):
        """Test the get_context method."""
        context = MaintenanceBot.get_context(
            self.bot.index, user_input="What is the capital of France?", top_k=2
        )
        self.assertIsInstance(context, str)
        self.assertTrue(context)

    def test_get_prompt_template(self):
        """Test the get_prompt_template method."""
        context = MaintenanceBot.get_context(
            self.bot.index, user_input="What is the capital of France?", top_k=2
        )
        prompt = self.bot.get_prompt_template(context)
        self.assertIsInstance(prompt, str)
        self.assertTrue(prompt)

    def test_inference(self):
        """Test the inference method."""
        response = self.bot.inference(user_input="What is the capital of France?")
        self.assertIsInstance(response, str)
        self.assertTrue(response)


if __name__ == "__main__":
    main()
