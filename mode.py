from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import spacy
from transformers import pipeline
import numpy as np
from sklearn.neighbors import NearestNeighbors
from crewai import Agent, Task, Crew, Process


@dataclass
class Order:
    order_id: str
    customer_id: str
    status: str
    items: List[Dict]
    created_at: datetime
    estimated_delivery: datetime


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    purchase_history: List[Dict]
    browsing_history: List[Dict]


class CustomerSupportChatbot:
    def __init__(self):
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.question_answerer = pipeline("question-answering")
        self.intent_classifier = pipeline("text-classification",
                                          model="distilbert-base-uncased-finetuned-sst-2-english")

        # Initialize recommendation system
        self.recommendation_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')

        # Store conversation context
        self.conversation_history = {}

        # Initialize Crew AI agents
        self._initialize_crew_agents()

    def _initialize_crew_agents(self):
        """Initialize specialized Crew AI agents for different tasks"""
        self.customer_service_agent = Agent(
            role='Customer Service Representative',
            goal='Handle customer inquiries and provide excellent service',
            backstory="""You are an experienced customer service representative 
                        with extensive knowledge of e-commerce operations.""",
            tools=[self._handle_general_query, self._classify_intent]
        )

        self.order_tracking_agent = Agent(
            role='Order Management Specialist',
            goal='Track and manage customer orders efficiently',
            backstory="""You specialize in order management systems and 
                        can quickly resolve order-related issues.""",
            tools=[self._handle_order_tracking, self._get_order_details]
        )

        self.returns_agent = Agent(
            role='Returns Specialist',
            goal='Process returns and refunds efficiently',
            backstory="""You are an expert in handling returns and refunds,
                        ensuring customer satisfaction while following company policies.""",
            tools=[self._handle_return_request, self._check_return_eligibility]
        )

        self.recommendation_agent = Agent(
            role='Product Recommendation Specialist',
            goal='Provide personalized product recommendations',
            backstory="""You excel at understanding customer preferences and
                        suggesting relevant products.""",
            tools=[self._handle_product_recommendation, self._generate_recommendations]
        )

    def process_message(self, customer_id: str, message: str) -> str:
        """Main entry point for processing customer messages using Crew AI"""
        # Add message to conversation history
        self._update_conversation_history(customer_id, message)

        # Create tasks based on intent
        intent = self._classify_intent(message)
        tasks = self._create_tasks(customer_id, message, intent)

        # Create and run crew
        crew = Crew(
            agents=[
                self.customer_service_agent,
                self.order_tracking_agent,
                self.returns_agent,
                self.recommendation_agent
            ],
            tasks=tasks,
            process=Process.sequential
        )

        result = crew.kickoff()
        return self._format_response(result)

    def _create_tasks(self, customer_id: str, message: str, intent: str) -> List[Task]:
        """Create appropriate tasks based on message intent"""
        tasks = []

        # Always add initial customer service task
        tasks.append(Task(
            description=f"Analyze customer message: {message}",
            agent=self.customer_service_agent
        ))

        # Add intent-specific tasks
        if intent == "order_tracking":
            tasks.append(Task(
                description=f"Track order for customer {customer_id}",
                agent=self.order_tracking_agent,
                context={"message": message, "customer_id": customer_id}
            ))

        elif intent == "return_request":
            tasks.append(Task(
                description=f"Process return request for customer {customer_id}",
                agent=self.returns_agent,
                context={"message": message, "customer_id": customer_id}
            ))

        elif intent == "product_recommendation":
            tasks.append(Task(
                description=f"Generate product recommendations for customer {customer_id}",
                agent=self.recommendation_agent,
                context={"message": message, "customer_id": customer_id}
            ))

        return tasks

    def _format_response(self, crew_result: str) -> str:
        """Format the crew's response into a customer-friendly message"""
        # Clean up and format the crew's response
        response = crew_result.strip()

        # Add sentiment-appropriate closing
        sentiment = self.sentiment_analyzer(response)[0]
        if sentiment['label'] == 'POSITIVE':
            response += "\n\nIs there anything else I can help you with?"
        elif sentiment['label'] == 'NEGATIVE':
            response += "\n\nI apologize if I couldn't fully address your concerns. Would you like to speak with a human representative?"

        return response

    # [Previous helper methods remain the same]
    def _classify_intent(self, message: str) -> str:
        """Classify the intent of the customer's message"""
        doc = self.nlp(message)

        # Extract key entities and patterns
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Rule-based intent classification
        if any(keyword in message.lower() for keyword in ["track", "order", "delivery", "shipped"]):
            return "order_tracking"
        elif any(keyword in message.lower() for keyword in ["recommend", "suggest", "similar"]):
            return "product_recommendation"
        elif any(keyword in message.lower() for keyword in ["return", "refund", "exchange"]):
            return "return_request"

        return "general_query"

    def _handle_order_tracking(self, customer_id: str, message: str) -> str:
        """Handle order tracking requests"""
        # Extract order ID from message using NER
        doc = self.nlp(message)
        order_id = None
        for ent in doc.ents:
            if ent.label_ == "ORDER_ID":
                order_id = ent.text

        if not order_id:
            return "Could you please provide your order number?"

        # Fetch order details from database (mock implementation)
        order = self._get_order_details(order_id)
        if not order:
            return "I couldn't find that order. Please check the order number and try again."

        return (f"Your order #{order_id} is currently {order.status}. "
                f"Estimated delivery: {order.estimated_delivery.strftime('%Y-%m-%d')}")

    # [Rest of the previous helper methods remain the same]