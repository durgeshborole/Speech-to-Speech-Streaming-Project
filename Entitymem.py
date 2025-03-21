import os
import dotenv
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationEntityMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain

dotenv.load_dotenv()

# Environment variables
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google GenAI LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7, api_key=GOOGLE_API_KEY)

# Initialize conversation memory
memory = ConversationEntityMemory(llm=llm)

# Define prompt template for innovation project ideas
TEMPLATE = """
You are an Innovation project idea AI, and your task is to give ideas based on the user's prompt.
You have to give optimal and optimized solutions that can be useful in today's world, along with the prerequisites required for the idea.

Steps to give the answer:
1. Understand the prompt.
2. The answer should be useful in today's world.
3. Include a detailed description of the solution.
4. Provide the necessary prerequisites or requirements to implement the idea.
5. Suggest possible challenges or obstacles one might face in implementing the idea.

Conversation History:
{entity}

User's Request:
{input}

Solution:
"""

# Create a prompt template instance
prompt = PromptTemplate.from_template(TEMPLATE)

# Initialize the conversation chain using the memory and prompt
conversation_chain = ConversationChain(
    llm=llm, 
    memory=memory,
    prompt=prompt
)

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
    response = conversation_chain.run(user_input)

    # Print the AI's response
    print("AI:", response)
