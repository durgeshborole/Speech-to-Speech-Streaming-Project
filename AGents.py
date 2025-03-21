import os
import dotenv
import ast
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
dotenv.load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Language Model (LLM)
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Define a safe calculator tool
class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Performs basic mathematical calculations. Input should be a valid mathematical expression."

    def _run(self, query: str) -> str:
        try:
            # Safely evaluate the mathematical expression
            result = eval(compile(ast.parse(query, mode='eval'), '<string>', 'eval'))
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)

# Define a weather tool (mock data)
class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = "Provides the current weather for a given city."

    def _run(self, location: str) -> str:
        weather_data = {
            "delhi": "Partly cloudy, 32°C with 60% humidity.",
            "mumbai": "Light rains, 28°C with 85% humidity.",
            "bangalore": "Sunny and pleasant, 25°C with 50% humidity.",
            "chennai": "Hot and humid, 35°C with 70% humidity.",
            "kolkata": "Cloudy, 30°C with 75% humidity.",
            "jaipur": "Sunny and hot, 38°C with 45% humidity.",
            "hyderabad": "Partly cloudy, 33°C with 55% humidity.",
            "pune": "Pleasant, 27°C with 65% humidity.",
            "ahmedabad": "Hot and dry, 40°C with 30% humidity.",
            "lucknow": "Humid, 34°C with 70% humidity."


        }
        return weather_data.get(location.lower().strip(), "Weather data unavailable. (Mock data)")

    async def _arun(self, location: str) -> str:
        return self._run(location)

# Define a search tool (mock data)
class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Searches the web for information based on input queries."

    def _run(self, query: str) -> str:
        search_results = {
            "artificial intelligence": "AI is intelligence demonstrated by machines, different from human intelligence.",
            "python programming": "Python is a high-level, general-purpose programming language with a focus on readability.",
            "climate change": "Climate change refers to long-term shifts in temperatures and weather patterns due to human activities."
        }
        return search_results.get(query.lower().strip(), f"No direct match found for '{query}'. (Mock data)")

    async def _arun(self, query: str) -> str:
        return self._run(query)

# Create tools list
tools = [CalculatorTool(), WeatherTool(), SearchTool()]

# Initialize memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the AI Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=False  # Set to False for better error visibility
)

# Function to run the AI
def run_chat():
    print("AI Agent initialized. Type 'exit' to quit or 'reset memory' to clear conversation history.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Goodbye! Have a great day. 😊")
            break

        elif user_input.lower() == "reset memory":
            memory.clear()
            print("Memory cleared! Starting a new conversation.")
            continue

        try:
            print(f"User  Input: {user_input}")  # Debugging line
            response = agent.run(user_input)
            print("\nAI:", response)
        except Exception as e:
            print("\nError:", str(e))

# Run the chatbot
if __name__ == "__main__":
    run_chat()