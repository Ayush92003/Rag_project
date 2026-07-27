from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_docling.loader import DoclingLoader
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert, you have to summarize the pdf uploaded by user"),
        ("human","{data}")
    ]
)

data = DoclingLoader(file_path="documents/JD.pdf")
docs = data.load()

prompt = template.format_messages(data=docs)

model = ChatMistralAI(model = "mistral-small-2506")

response = model.invoke(prompt)

print(response.content)
