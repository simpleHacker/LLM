# https://www.stephendiehl.com/posts/faiss/
from sentence_transformers import SentenceTransformer
# can use local download model
model2 = SentenceTransformer("./models/all-mpnet-base-v2")
embeddings = model2.encode(sentences)
print("Embeddings: ", embeddings)

# set up FAISS to perform similary search, inputs has to be in form of dense vectors
import faiss
# demensions of our embeddings
d = embeddings.shape[1]
# create a index for our dense vectors, no do the chunking for this example, because just sample docs
index = faiss.IndexFlatL2(d) # using L2 (Euclidean) distance
# Adding the embeddings to the index
index.add(embeddings)
print(f"Total Sentences indexed: {index.ntotal}")

# perform a semantic search
query = "how many copy of bas schema in our system?"
query_embedding = model2.encode([query])

k = 1 # Number of nearest neighbors to retrieve

distances, indices = index.search(query_embedding, k)
# Display the results
print(f"Query: {query}")
print("Answers:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}: {sentences[idx]} (Distance: {distances[0][i]})")

########## Second approach ##############
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# load the document
raw_documents = TextLoader('example.txt').load()
# splitter to chunking the documents
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator = "\n",)
# chunked document
documents = text_splitter.split_documents(raw_documents)
# Create a dictionary with model configuration options, specifying to use the CPU for computations
modelPath = "./models/all-mpnet-base-v2"

model_kwargs = {'device' : 'cpu'}
# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': True}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)
# load in vector store of the chunked documents' embeddings
db = FAISS.from_documents(documents, embeddings)

query = "what cannot we do to schema file?"
docs = db.similarity_search(query, k=2)
for doc in docs:
    print(doc.page_content)

############ Combine with LLM ################
# combine RAG with LLM
## build the context with close retrieval result
context = "\n\n".join(x.page_content for x in docis)
## build prompt template
system_message = "You are a professional and technical assistant trained to answer questions about {domain}, {domain_desc} \n\nPlease answer your questions closely based on below: {context}"
