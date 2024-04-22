from unittest import TestCase, main
from time import time

from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.picker_bot import MaintenanceBot, PromptTemplate

start_time = time()
bot = MaintenanceBot()
print(f"Time to load the bot: {time() - start_time:.2f} seconds")

USER_INPUT = "How do I fix this apple picking robot?"


class TestBot(TestCase):
    """Test the MaintenanceBot class."""

    def test_post_init_steps(self):
        """Test the get_context method."""
        self.assertTrue(bot.dataset_path.exists())
        self.assertIsNotNone(bot.model)
        self.assertIsNotNone(bot.tokenizer)
        self.assertIsNotNone(bot.index)
        self.assertIsInstance(bot.text_splitter, RecursiveCharacterTextSplitter)
        self.assertIsInstance(bot.index.vectorstore, InMemoryVectorStore)

    def test_get_gps_nav_context(self):
        """Test the get_context method."""
        user_input = "I'm lost in the woods. How do I get back to civilization?"
        context = MaintenanceBot.get_context(
            bot.vectorstore, user_input=user_input, top_k=1
        )
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        self.assertIn("GPS", context)

    def test_get_fruit_spill_context(self):
        user_input = "my fruit is spilling all over the place. what should I do?"
        context = MaintenanceBot.get_context(
            bot.vectorstore, user_input=user_input, top_k=1
        )
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        self.assertIn("overfilling", context)

    def test_get_prompt_template(self):
        """Test the get_prompt_template method."""
        context = MaintenanceBot.get_context(
            bot.index.vectorstore, user_input=USER_INPUT, top_k=2
        )
        prompt = bot.get_prompt_template(context)
        print(prompt)
        self.assertIsInstance(prompt, PromptTemplate)
        self.assertTrue(str(prompt))

    def test_inference(self):
        """Test the inference method."""
        response = bot.inference(USER_INPUT)
        print(response)
        print(next(response))
        print(next(response))
        print(next(response))
        print(next(response))
        print(next(response))
        print(next(response))
        print(next(response))
        print(next(response))
        print(next(response))
        self.assertIsNotNone(response)
        self.assertTrue(response)


if __name__ == "__main__":
    main()
