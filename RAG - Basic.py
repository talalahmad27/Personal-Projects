import os
import ollama
import chromadb

docs = []
#curr_filename = "F:\\Macquarie\\Research project\\RAG\\files"
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'files')
for filename in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'files')):
    print(filename)
    with open(os.path.join(file_path,filename), 'r') as file:
        docs.append(file.read())
        

client = chromadb.Client()
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(docs):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )

    
# an example prompt
prompt = "What is Covid?"

#generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
    prompt=prompt,
    model="mxbai-embed-large")

results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)

data = results['documents'][0][0]

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
    model="llama2",
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])
