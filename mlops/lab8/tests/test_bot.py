from logging import debug
from unittest import TestCase, main
from time import time

from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch import set_default_dtype, float32

from model.picker_bot import MaintenanceBot, PromptTemplate



class _TestBot(TestCase):
    """Test the MaintenanceBot class."""

    def setUp(self) -> None:
        self.USER_INPUT = "How do I fix this apple picking robot?"
        set_default_dtype(float32)
        self.niters = 10
        start_time = time()
        self.bot = MaintenanceBot()
        debug(f"Time to load the bot: {time() - start_time:.2f} seconds")
        return super().setUp()



class TestBotContextMethod(_TestBot):
    def test_get_gps_nav_context(self):
        """Test the get_context method."""
        user_input = "I'm lost in the woods. How do I get back to civilization?"
        context = MaintenanceBot.get_context(
            self.bot.vectorstore, user_input=user_input, top_k=1
        )
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        self.assertIn("GPS", context)

    def test_get_fruit_spill_context(self):
        user_input = "my fruit is spilling all over the place. what should I do?"
        context = MaintenanceBot.get_context(
            self.bot.vectorstore, user_input=user_input, top_k=1
        )
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        self.assertIn("overfilling", context)




class TestBotMethods(_TestBot):

    def test_post_init_steps(self):
        """Test the get_context method."""
        self.assertTrue(self.bot.dataset_path.exists())
        self.assertIsNotNone(self.bot.model)
        self.assertIsNotNone(self.bot.tokenizer)
        self.assertIsNotNone(self.bot.index)
        self.assertIsInstance(self.bot.text_splitter, RecursiveCharacterTextSplitter)
        self.assertIsInstance(self.bot.index.vectorstore, InMemoryVectorStore)

    def test_get_prompt_template(self):
        """Test the get_prompt_template method."""
        context = MaintenanceBot.get_context(
            self.bot.index.vectorstore, user_input=self.USER_INPUT, top_k=2
        )
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        prompt = self.bot.get_prompt_template(context)
        self.assertIsInstance(prompt, PromptTemplate)
        self.assertTrue(str(prompt))

    def test_inference(self):
        """Test the inference method."""
        response = self.bot.inference(self.USER_INPUT)
        self.assertIsNotNone(response)
        for _ in range(self.niters):
            self.assertIsInstance(next(response), str)


if __name__ == "__main__":
    main()
