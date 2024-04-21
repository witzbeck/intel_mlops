from os import environ
from pathlib import Path
from unittest import main, TestCase

from app.__init__ import here
from model.picker_bot import DATASET_NAME, DATASET_PATH, MODEL_NAME, TOKEN


class TestInitConditions(TestCase):
    """Test the initialization conditions."""

    def test_here_is_path(self):
        """Test that here is a Path object and exists."""
        self.assertIsInstance(here, Path)
        self.assertTrue(here.exists())

    def test_model_name(self):
        """Test that MODEL_NAME is set in the environment."""
        self.assertIsNotNone(MODEL_NAME)

    def test_token(self):
        """Test that TOKEN is set in the environment."""
        self.assertIsNotNone(TOKEN)

    def test_DATASET_NAME(self):
        """Test the DATASET_NAME variable."""
        self.assertIsNotNone(DATASET_NAME)

    def test_DATASET_PATH(self):
        """Test the DATASET_PATH variable."""
        self.assertIsNotNone(DATASET_PATH)
        self.assertIsInstance(DATASET_PATH, Path)
        self.assertTrue(DATASET_PATH.exists())

    def test_fastapi_envs(self):
        """Test the FastAPI environment variables."""

        self.assertIn("HF_MODEL", environ)
        self.assertIn("LOG_LEVEL", environ)
        self.assertIn("WORKERS_PER_CORE", environ)
        self.assertIn("TIMEOUT", environ)

    def test_get_context(self):
        """Test the get_context method."""
        pass

    def test_get_prompt_template(self):
        """Test the get_prompt_template method."""
        pass

    def test_inference(self):
        """Test the inference method."""
        pass


if __name__ == "__main__":
    main()
