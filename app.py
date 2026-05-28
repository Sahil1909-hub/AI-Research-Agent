from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
import os
from langchain.tools import tool

load_dotenv()


api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=api_key
)


serper_api_key = os.getenv('SERPER_API_KEY')


search = GoogleSerperAPIWrapper()


@tool
def web_search(query:str):
    """
    this tool returns answer for given user query.
    """
    return search.run(query)



system_prompt = "You are an helful agent, answer the user query with the given search tool." 

search_agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt=system_prompt
)



query = "what is 3000-2999+1?"


response = search_agent.invoke({"messages": [{
    "role":"user",
    "content":query
}]})


answer = response['messages'][-1].content
print(answer)