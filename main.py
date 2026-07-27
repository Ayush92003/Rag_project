from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_docling.loader import DoclingLoader
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert who summarizes text in json format given by user"),
        ("human","{data}")
    ]
)

model = ChatMistralAI(model = "mistral-small-2506")

data = DoclingLoader(file_path="documents/notes.txt")
docs = data.load()

prompt = template.format_messages(data = docs[0].page_content)

response = model.invoke(prompt)

print(response.content)