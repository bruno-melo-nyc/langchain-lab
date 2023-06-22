from langchain.chains import RetrievalQA
#from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader # used to load proprietary data that we will combine with the LLM
from langchain.vectorstores import DocArrayInMemorySearch # It's in memory, so no need to connect to external database
from IPython.display import display, Markdown
from langchain.llms import HuggingFaceHub
from langchain.indexes import VectorstoreIndexCreator ## Helps us create a vector store really easily. Easy because it's in-memory
from langchain.embeddings import HuggingFaceEmbeddings
import os
import getpass


HUGGINGFACEHUB_API_TOKEN="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"

os.environ['HUGGINGFACEHUB_API_TOKEN'] = getpass.getpass(HUGGINGFACEHUB_API_TOKEN)

#hf = HuggingFaceHub(repo_id="gpt2", huggingfacehub_api_token="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")
#from huggingface_hub import login
#login("HUGGINGFACEHUB_API_TOKEN")



repo_id = "google/flan-t5-base"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN , model_kwargs={"temperature": 0, "max_length": 64})

embeddings = HuggingFaceEmbeddings()

file = 'C:/Users/bsverasdemel/OneDrive - New York Life/Risk Coverage/Research/LLM/msEuropeanETFs.csv'

loader = CSVLoader(file_path=file)
#index_creator = VectorstoreIndexCreator(embedding = embeddings)
#searcher = DocArrayInMemorySearch(index_creator,embedding = embeddings).from_loaders([loader])

## vector store class
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch, embedding = embeddings).from_loaders([loader])

## Now the vector store has been created! And we can ask questions about it.

query ="What is the oldest fund"

response = index.query(query)