from langchain.chains.conversation.base import ConversationChain
import os
import dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
dotenv.load_dotenv()

# Get model and API key from environment variables
GEMINI_MODEL = os.getenv("GEMINI_MODEL")  # e.g., gemini-1.5-flash
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Your Google API key

# Initialize the language model
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Initialize memory for the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt template
TEMPLATE = """
You are an Shayari AI, and your task is to give shayari's based on the user's prompt.
it should include:
1.it should be creative ie. it should not be common
2.It should have rhyme.
3.It should have rommance.
You have to give optimal and optimized solutions that can be useful in today's world, along with the prerequisites required for the idea.

Steps to give the answer:
1. Understand the prompt.
2. The answer should be useful in today's world.
Conversation History:
{chat_history}

:User  {input}
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
    print("AI:", response)

# import os
# import dotenv
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
#
# dotenv.load_dotenv()
#
# GEMINI_MODEL = os.getenv("GEMINI_MODEL") # gemini-1.5-flash
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # A3zaSyA2ljk9PrtRZ-MJswTLFWdbsl5G2O0M7Eo
#
# llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)
#
# TEMPLATE = """
# You are an Shayari AI, and your task is to give shayari's based on the user's prompt.
#  it should include:
#  1.it should be creative ie. it should not be common
#  2.It should have rhyming words at the end of the sentence
#  3.It should have sarcasm.
#
# {input}
# """
#
# prompt = PromptTemplate.from_template(TEMPLATE)
#
# chain = prompt | llm
# while True:
#     user_input = input("\nYou: ")
#
#     if user_input.lower() == "exit":
#         print("Goodbye! Have a great day.")
#         break
#
#     elif user_input.lower() == "reset memory":
#         print("Memory cleared! Starting a new conversation.")
#         continue
#
#     # Get the response from the conversation chain
#     response = chain.invoke({"input": user_input})
#
#     # Print the response
#     print("AI:", response)