# from dotenv import load_dotenv
# import os

# load_dotenv()
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate

# if __name__ == "__main__":
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#     query = "What is ai"

#     vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"),embedding=embeddings)

#     prompt = hub.pull('langchain-ai/retrieval-qa-chat')

#     combined_docs_chain = create_stuff_documents_chain(llm,prompt)

#     retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),combine_docs_chain=combined_docs_chain)

#     result = retrieval_chain.invoke({"input": query})

#     print(result)


# from dotenv import load_dotenv
# import os
# from langchain import hub
# from langchain.chains import create_retrieval_chain
# from langchain_pinecone import PineconeVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain

# # Load environment variables
# load_dotenv()

# # Function to parse and format the output to give only the relevant answer
# def parse_output(result):
#     """
#     Process the output to ensure only the relevant information is returned.
#     Remove any preambles or unnecessary context.
#     """
#     try:
#         # Assuming result is a string, strip whitespace and handle unwanted context
#         cleaned_result = result.strip()

#         # If the result starts with a preamble like 'Here's the answer:', remove it
#         if cleaned_result.lower().startswith("here's the answer"):
#             cleaned_result = cleaned_result[len("here's the answer:"):].strip()

#         # Return the cleaned result
#         return cleaned_result
#     except Exception as e:
#         return f"Error parsing output: {e}"


# if __name__ == "__main__":
#     # Initialize embeddings and language model (LLM)
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#     query = "What is AI?"  # Example query

#     # Set up the Pinecone vector store (make sure the index is created in Pinecone)
#     vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

#     # Pull the prompt template for document retrieval and QA
#     prompt = hub.pull('langchain-ai/retrieval-qa-chat')

#     # Create the chain for combining documents using the LLM and retrieved docs
#     combined_docs_chain = create_stuff_documents_chain(llm, prompt)

#     # Create the retrieval chain to fetch relevant documents from Pinecone
#     retrieval_chain = create_retrieval_chain(
#         retriever=vectorstore.as_retriever(),
#         combine_docs_chain=combined_docs_chain
#     )

#     # Retrieve the documents based on the query
#     result = retrieval_chain.invoke({"input": query})

#     # Parse and refine the result (make it readable and coherent)
#     refined_result = parse_output(result)

#     # Print the specific refined result related to the query
#     print(refined_result)

from dotenv import load_dotenv
import os
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Function to refine and clean the output (get only the relevant part)
def refine_output(result):
    """
    Process the result and return only the relevant answer from the retrieved documents.
    Removes preambles or any unnecessary context.
    """
    try:
        # Check if result is a dictionary, and extract the 'output' key
        if isinstance(result, dict):
            answer = result.get('output', '').strip()
        else:
            # If the result is not in the expected format, return as is
            answer = str(result).strip()

        # Debugging: Print the raw result to inspect the response format  # Print raw result here
        
        # Clean or remove any unwanted phrases (you can extend this to more patterns)
        if answer.lower().startswith("answer:"):
            answer = answer[len("answer:"):].strip()
        elif answer.lower().startswith("here's the information:"):
            answer = answer[len("here's the information:"):].strip()

        # Return the cleaned, relevant answer
        return answer
    
    except Exception as e:
        return f"Error refining output: {e}"

if __name__ == "__main__":
    # Initialize embeddings and language model (LLM)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    query = "What is AI?"  # Example query

    # Set up the Pinecone vector store (make sure the index is created in Pinecone)
    vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

    # Pull the prompt template for document retrieval and QA
    prompt = hub.pull('langchain-ai/retrieval-qa-chat')

    # Create the chain for combining documents using the LLM and retrieved docs
    combined_docs_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain to fetch relevant documents from Pinecone
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combined_docs_chain
    )

    # Retrieve the documents based on the query
    result = retrieval_chain.invoke({"input": query})
    # Refine and clean the result to return only the relevant part
    refined_result = refine_output(result)

    # Print the specific refined result related to the query
    print("Refined Output:", refined_result)
