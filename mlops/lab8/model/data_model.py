from functools import partial
from random import choice
from pydantic import BaseModel

url_base = "http://127.0.0.1:80"
headers = {"Content-Type": "application/json"}


class GenPayload(BaseModel):
    user_input: str


TEMPLATE_BASE = """
        Please use the following apple picker technical support related questions to answer questions. 
        Context: {context}
        ---
        This is the user's question: {question}
        Answer: This is what our auto apple picker technical expert suggest."""
COUNTRIES = [
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
get_country = partial(choice, COUNTRIES)
