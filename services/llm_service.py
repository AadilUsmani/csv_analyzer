import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict

load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
)

DEPLOYMENT_NAME = "gpt-4o-mini"

# Default prompt template
PROMPT_TEMPLATE = """
You are a professional data analyst. You are given a CSV dataset context and a user query.

Steps:
1. Analyze the dataset and figure out what information is most relevant.
2. Think step by step about what the query is asking.
3. Provide a final clear answer that is useful and well-structured.

Only output the final answer, not your reasoning.

Dataset Context:
{context}

User Query:
{query}

Final Answer:
"""

# Shared conversation memory
chat_history: List[Dict[str, str]] = []


def summarize_history(history_text: str) -> str:
    """Summarize long conversation history using the LLM."""
    summary_prompt = f"Summarize this conversation briefly:\n{history_text}"
    summary_response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.2
    )
    return summary_response.choices[0].message.content.strip()


def query_csv_with_llm(
    df: pd.DataFrame,
    summary: dict,
    user_query: str,
    prompt_template: str = PROMPT_TEMPLATE,
    max_history_tokens: int = 2000
) -> str:
    """
    Query the LLM with a prompt template, keep chat history,
    and summarize history when it becomes too long.
    """
    schema_info = {"columns": list(df.columns), "rows": len(df)}
    context = {"summary": summary, "schema": schema_info}

    # Add user query to history
    chat_history.append({"role": "user", "content": user_query})

    # Summarize if history grows too long
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
    if len(history_text.split()) > max_history_tokens:
        summarized = summarize_history(history_text)
        chat_history.clear()
        chat_history.append({"role": "system", "content": f"Conversation summary: {summarized}"})

    # Format the prompt
    filled_prompt = prompt_template.format(context=context, query=user_query)

    messages = [
        {"role": "system", "content": "You are a data analysis assistant. Answer based on CSV context."},
        *chat_history,  # include conversation history
        {"role": "user", "content": filled_prompt}
    ]

    # Query LLM
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.2
    )

    reply = response.choices[0].message.content.strip()
    chat_history.append({"role": "assistant", "content": reply})

    return reply
