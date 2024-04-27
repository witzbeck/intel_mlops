from pydantic import BaseModel

url_base = "http://127.0.0.1:80"
headers = {"Content-Type": "application/json"}

class GenPayload(BaseModel):
    data: str
    user_input: str


TEMPLATE_BASE = """
        Please use the following apple picker technical support related questions to answer questions. 
        Context: {context}
        ---
        This is the user's question: {question}
        Answer: This is what our auto apple picker technical expert suggest."""