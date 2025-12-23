
#requires langchain 0.3  disreagard if using langchain 1.x
from dotenv import load_dotenv
#from langchain.agents import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, tool

import os


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)




if __name__ == "__react_pt1__":
    print("Hello LangChain Tools ")
    tools = [get_text_length] #supplied to React Agent

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer\n
    Thought: you should always think about what to do\n
    Action: the action to take, should be one of [{tool_names}]\n
    Action Input: the input to the action\n
    Observation: the result of the action\n
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer\n
    Final Answer: the final answer to the original input question\n
    \n
    Begin!

    Question: {input}
    Thought:
    """

    #prompt = PromptTemplate.from_template(template=template).partial(tools=tools, tool_names=", ".join([t.name for t in tools]))
    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools), 
    tool_names=", ".join([t.name for t in tools]))

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0, 
        model_kwargs={"stop": ["\nObservation", "Observation"]}, # stopping criteria
        ) #model_kwargs to control stopping criteria, due to gemini llm behavior, not needed for openai
    agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()

    res = agent.invoke({"input": "What is the length of 'DOG' in characters?'"})
    print(f"Final Answer: {res}")
