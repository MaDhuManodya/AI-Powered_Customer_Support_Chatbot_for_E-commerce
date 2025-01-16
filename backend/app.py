# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn
from crewai import Agent, Task, Crew, Process
import spacy
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(title="Customer Support Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class Message(BaseModel):
    customer_id: str
    content: str
    timestamp: datetime = datetime.now()


class ChatResponse(BaseModel):
    response: str
    intent: str
    suggested_actions: List[str]
    timestamp: datetime = datetime.now()


# Initialize NLP components
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")

# Initialize Crew AI agents
customer_service_agent = Agent(
    role='Customer Service Representative',
    goal='Handle customer inquiries and provide excellent service',
    backstory="Experienced customer service representative with e-commerce knowledge."
)

order_tracking_agent = Agent(
    role='Order Management Specialist',
    goal='Track and manage customer orders efficiently',
    backstory="Order management specialist with deep system knowledge."
)

# In-memory store for conversations
conversations = {}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: Message):
    try:
        # Process message
        intent = classify_intent(message.content)
        response = process_message(message.content, intent)

        # Store conversation
        if message.customer_id not in conversations:
            conversations[message.customer_id] = []

        conversations[message.customer_id].append({
            "content": message.content,
            "timestamp": message.timestamp,
            "type": "user"
        })

        # Create response
        chat_response = ChatResponse(
            response=response,
            intent=intent,
            suggested_actions=get_suggested_actions(intent),
            timestamp=datetime.now()
        )

        # Store bot response
        conversations[message.customer_id].append({
            "content": response,
            "timestamp": chat_response.timestamp,
            "type": "bot"
        })

        return chat_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversation/{customer_id}")
async def get_conversation(customer_id: str):
    return {"history": conversations.get(customer_id, [])}


def classify_intent(message: str) -> str:
    doc = nlp(message.lower())

    if any(keyword in doc.text for keyword in ["track", "order", "delivery"]):
        return "order_tracking"
    elif any(keyword in doc.text for keyword in ["return", "refund"]):
        return "return_request"
    elif any(keyword in doc.text for keyword in ["recommend", "suggest"]):
        return "product_recommendation"

    return "general_query"


def process_message(message: str, intent: str) -> str:
    tasks = [
        Task(
            description=f"Process customer message: {message}",
            agent=customer_service_agent
        )
    ]

    if intent == "order_tracking":
        tasks.append(
            Task(
                description="Track order status",
                agent=order_tracking_agent
            )
        )

    crew = Crew(
        agents=[customer_service_agent, order_tracking_agent],
        tasks=tasks,
        process=Process.sequential
    )

    return crew.kickoff()


def get_suggested_actions(intent: str) -> List[str]:
    suggestions = {
        "order_tracking": ["Check order status", "Contact shipping", "View delivery details"],
        "return_request": ["Start return", "Print label", "View policy"],
        "product_recommendation": ["View similar", "Check reviews", "Add to cart"],
        "general_query": ["Browse FAQ", "Contact support", "View account"]
    }
    return suggestions.get(intent, ["Contact support", "Browse FAQ"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)