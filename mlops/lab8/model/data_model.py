from pydantic import BaseModel

url_base = "http://localhost:8000"
headers = {"Content-Type": "application/json"}


class GenPayload(BaseModel):
    data: str
    user_input: str
