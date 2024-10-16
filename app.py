from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline

# Load T5-small model and tokenizer
model_name = "google-t5/t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Create a text generation pipeline
text_generation_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7
)

# Create a LangChain LLM from the pipeline
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Load and process documents from a local file
loader = TextLoader("./NeuralNetworkWikipedia.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings using a smaller model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
db = Chroma.from_documents(texts, embeddings)

# Create a retriever
retriever = db.as_retriever()

# Create a prompt template
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Example query
query = "What is an artificial neuron?"
result = qa_chain({"query": query})

print("Question:", query)
print("Answer:", result["result"])