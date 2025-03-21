from langchain.chains.conversation.base import ConversationChain
import os
import dotenv
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables
dotenv.load_dotenv()

# Get model and API key from environment variables
GEMINI_MODEL = os.getenv("GEMINI_MODEL")  # e.g., gemini-1.5-flash
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Your Google API key

# Initialize the language model
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Initialize the vector store (e.g., using FAISS)
# Assuming you have pre-loaded or have data to populate the FAISS index
embedding = OpenAIEmbeddings()  # You can use any embedding model
vector_store = FAISS.load_local("faiss_index", embedding)  # Load your existing FAISS index (or create one)

# Initialize VectorStoreRetrieverMemory with the vector store
memory = VectorStoreRetrieverMemory(vectorstore=vector_store)

# Define the prompt template
TEMPLATE = """
You are an Innovation project idea AI, and your task is to give ideas based on the user's prompt.
You have to give optimal and optimized solutions that can be useful in today's world, along with the prerequisites required for the idea.

Steps to give the answer:
1. Understand the prompt.
2. The answer should be useful in today's world.

Conversation History:
{chat_history}

:User      {input}
"""

# Create the prompt template
prompt = PromptTemplate.from_template(TEMPLATE)

# Create the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# Start the conversation loop
while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Goodbye! Have a great day.")
        break

    elif user_input.lower() == "reset memory":
        memory.clear()
        print("Memory cleared! Starting a new conversation.")
        continue

    # Get the response from the conversation chain
    response = conversation({"input": user_input})

    # Print the response
    print("AI:", response['output'])  # Correctly access the output
