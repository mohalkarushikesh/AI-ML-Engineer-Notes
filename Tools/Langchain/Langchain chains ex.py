from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv(dotenv_path="./.env")

# Access the API key from the environment
api_key = os.getenv("GOOGLE_GEN_API")

# Initialize the LLM with the API key
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

"""
# Create a prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You will provide information about {topic}"),
        ("human", "tell me visiting place, best foods, culture, education of {city}"),
    ]
)

# Create chain of prompt template, LLM & output parser
Chain = prompt_template | llm | StrOutputParser()

# Invoke the user queries that are needed for the prompt template and generate output
result = Chain.invoke({"topic": "place", "city": "Mumbai"})
print(result)

"""

"""

# Alternative implementation using RunnableSequence

from langchain_core.runnables import RunnableLambda, RunnableSequence

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(response)
"""


"""

# Alternative implementation with more detailed stepss

from langchain.schema import HumanMessage, SystemMessage

# Step 1: Format the prompt (input is a dictionary with topic and count)
format_prompt = RunnableLambda(lambda x: f"Generate {x['joke_count']} jokes about {x['topic']}.")

# Step 2: Keyword checker (check if the topic is valid, else raise an error)
check_topic = RunnableLambda(lambda x: x if x.lower().find("lawyers") != -1 else "Invalid topic. Use 'lawyers'.")

# Step 3: Convert the prompt to uppercase
uppercase_prompt = RunnableLambda(lambda x: x.upper())

# Step 4: Add an extra phrase to the prompt (e.g., "Let's get started with some jokes!")
add_phrase = RunnableLambda(lambda x: x + " LET'S GET STARTED WITH SOME JOKES!")

# Step 5: Convert the string into a message format for the model
to_messages = RunnableLambda(lambda x: [SystemMessage(content="Tell jokes"), HumanMessage(content=x)])

# Step 6: Dummy model invocation (Here, we simulate a model's response as a string)
invoke_model = RunnableLambda(lambda x: llm.invoke(x))

# Step 7: Parse the output from the model's response
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence
chain = RunnableSequence(
    first=format_prompt,
    middle=[check_topic, uppercase_prompt, add_phrase, to_messages, invoke_model],
    last=parse_output
)

# Input for the chain
input_data = {"topic": "lawyers", "joke_count": 3}

# Run the chain
response = chain.invoke(input_data)

# Output the response
print(f"Response from the AI: {response}")

"""

"""
# Extended chain with additional processing steps

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | llm | StrOutputParser() | uppercase_output | count_words

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)

"""


"""

# Parallel processing within a chain

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)


# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features.",
            ),
        ]
    )
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features.",
            ),
        ]
    )
    return cons_template.format_prompt(features=features)


# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | llm | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | llm | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | llm
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Run the chain
result = chain.invoke({"product_name": "MacBook Pro"})

# Output
print(result)

"""


# Branching logic based on feedback sentiment


from langchain_core.runnables import RunnableBranch

# Define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | llm | StrOutputParser()  # Positive feedback chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | llm | StrOutputParser()  # Negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | llm | StrOutputParser()  # Neutral feedback chain
    ),
    escalate_feedback_template | llm | StrOutputParser()
)

# Create the classification chain
classification_chain = classification_template | llm | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "The product is terrible. It broke after just one use and the quality is very poor."
result = chain.invoke({"feedback": review})

# Output the result
print(result)

