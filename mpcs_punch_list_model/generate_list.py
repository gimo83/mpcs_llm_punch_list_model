import ollama
import chromadb
from chromadb.utils import embedding_functions
import os

# --- 1. Setup the Vector Database (Your AI's Memory) ---
chroma_client = chromadb.PersistentClient(path="./project_memory")
# Use a standard embedding model to convert text to numbers
embedding_func = embedding_functions.OllamaEmbeddingFunction(
    model_name="llama3.2:3b" # Run 'ollama pull nomic-embed-text' first
)

# Create or get a collection for your project data
collection = chroma_client.get_or_create_collection(
    name="historical_projects",
    embedding_function=embedding_func
)

# --- 2. Ingest Historical Data (Do this once) ---
def ingest_project_data(project_files_path):
    """Reads text files from a folder and stores them in ChromaDB."""
    # This is a simplified example. You'd loop through your docs here.
    # For each document, you'd create a chunk of text and an ID.
    for i, filename in enumerate(os.listdir(project_files_path)):
        with open(os.path.join(project_files_path, filename), 'r') as f:
            content = f.read()
            # Add the document content to the database
            collection.add(
                documents=[content],
                metadatas=[{"source": filename}],
                ids=[f"proj_{i}"]
            )
    print("Data ingestion complete!")

# --- 3. Query the Model with RAG (Your Main Function) ---
def generate_tasks(project_description):
    """Finds relevant historical projects and asks the AI to generate tasks."""
    
    # 1. Retrieve: Search the database for similar past projects
    results = collection.query(
        query_texts=[project_description],
        n_results=3 # Get the top 3 most similar projects
    )
    
    # 2. Augment: Combine the historical context with the new prompt
    historical_context = "\n---\n".join(results['documents'][0])
    
    full_prompt = f"""
    Based on the following historical project data:
    {historical_context}
    
    Create a task list for this new project:
    {project_description}
    """
    
    # 3. Generate: Use your custom Modelfile to create the response
    response = ollama.generate(
        model='mpcs-punch-list-model', # Your custom model from Step 1
        prompt=full_prompt,
        options={'temperature': 0.4} # Enforce low creativity for task lists
    )
    
    return response['response']

# --- Example Usage ---
# First, ingest your data (you only need to do this once)
# ingest_project_data("./historical_data/")

# Now, for every new project, just call this function
new_project = "Build a responsive e-commerce website with user login and payment gateway."
task_list = generate_tasks(new_project)
print(task_list)
