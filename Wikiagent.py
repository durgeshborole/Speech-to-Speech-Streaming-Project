import os
import dotenv
import ast
from langchain.agents import AgentType, initialize_agent
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
import wikipediaapi

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
            "london": "Partly cloudy, 68°F with 65% humidity.",
            "tokyo": "Light rains, 59°F with 80% humidity.",
            "new york": "Sunny and hot, 75°F with 55% humidity.",
            "chicago": "Windy, 77°F with 58% humidity."
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

# Define a Wikipedia tool using wikipedia-api
class WikipediaTool(BaseTool):
    name: str = "wikipedia"
    description: str = "Fetches summaries from Wikipedia based on a query."

    def _run(self, query: str) -> str:
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page = wiki_wiki.page(query)
        if page.exists():
            return page.summary
        else:
            return "No summary found for the given query."

    async def _arun(self, query: str) -> str:
        return self._run(query)

# Initialize memory for the agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent with the tools
tools = [CalculatorTool(), WeatherTool(), SearchTool(), WikipediaTool()]
agent = initialize_agent(tools=tools , llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory)


# Example usage of the agent
if __name__ == "__main__":
    user_input = "Tell me about artificial intelligence."
    response = agent.run(user_input)
    print(response)