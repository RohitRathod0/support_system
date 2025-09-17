from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.memory import ShortTermMemory, LongTermMemory
from typing import List, Dict, Any, Optional
import json
import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from datetime import datetime, timedelta
import uuid
import requests
from urllib.parse import quote
import time

# Alternative web search using requests instead of SerperDevTool
class SimpleWebSearchManager:
    """Simple web search manager without external API dependencies"""

    def __init__(self, search_api_key: str = None):
        self.search_api_key = search_api_key or os.getenv("SEARCH_API_KEY")
        self.search_cache = {}
        self.cache_duration = timedelta(hours=6)

    def should_search_web(self, query: str, internal_confidence: float) -> bool:
        """Determine if web search is needed based on query and internal confidence"""
        search_triggers = ["error", "bug", "crash", "not working", "issue", "down", 
                          "outage", "latest", "current", "today", "recently", "new"]

        query_lower = query.lower()
        has_triggers = any(trigger in query_lower for trigger in search_triggers)

        return has_triggers or internal_confidence < 0.7

    def perform_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Perform web search using simple approach (can be enhanced with actual APIs later)"""
        results = []
        for query in queries[:2]:  # Limit to 2 queries
            try:
                # Simple implementation - returns structured response
                # In production, replace with actual search API
                result = {
                    "query": query,
                    "content": f"Search results for: {query}. This is a placeholder response that would contain actual web search results.",
                    "source": "web_search",
                    "url": f"https://search.example.com?q={quote(query)}",
                    "title": f"Search: {query}"
                }
                results.append(result)
            except Exception as e:
                print(f"Web search error for '{query}': {str(e)}")

        return results

class CSVPolicyProcessor:
    """Processes company policy CSV files"""

    def __init__(self):
        self.processed_policies = {}

    def process_csv_file(self, csv_file_path: str) -> Dict[str, Any]:
        """Process CSV file and extract policy information"""
        try:
            if not os.path.exists(csv_file_path):
                return {}

            df = pd.read_csv(csv_file_path)
            policies = {}

            for _, row in df.iterrows():
                policy_id = row.get('policy_id', f"policy_{len(policies)}")
                policies[policy_id] = {
                    "category": row.get('category', 'general'),
                    "title": row.get('title', 'Untitled Policy'),
                    "description": row.get('description', ''),
                    "rules": row.get('rules', ''),
                    "authorization_level": row.get('authorization_level', 'agent'),
                    "created_date": datetime.now().isoformat()
                }

            self.processed_policies.update(policies)
            return policies

        except Exception as e:
            print(f"Error processing CSV file {csv_file_path}: {str(e)}")
            return {}

class VectorStoreManager:
    """Manages vector storage for all data types"""

    def __init__(self, db_path: str = "./vector_db"):
        self.db_path = db_path
        try:
            self.client = chromadb.PersistentClient(path=db_path)

            # Initialize embedding function
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-ada-002"
                )
            else:
                embedding_fn = embedding_functions.DefaultEmbeddingFunction()

            # Create collections
            self.user_collection = self.client.get_or_create_collection(
                name="user_profiles",
                embedding_function=embedding_fn
            )

            self.conversation_collection = self.client.get_or_create_collection(
                name="conversations",
                embedding_function=embedding_fn
            )

            self.policy_collection = self.client.get_or_create_collection(
                name="company_policies",
                embedding_function=embedding_fn
            )

        except Exception as e:
            print(f"Warning: Vector store initialization failed: {str(e)}")
            self.client = None

    def store_conversation(self, user_id: str, query: str, response: str, metadata: Dict = None):
        """Store conversation data"""
        if not self.client:
            return

        try:
            conv_text = f"User: {query} Response: {response}"
            conv_id = f"conv_{user_id}_{datetime.now().timestamp()}"

            self.conversation_collection.add(
                documents=[conv_text],
                metadatas=[{
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "response": response,
                    **(metadata or {})
                }],
                ids=[conv_id]
            )
        except Exception as e:
            print(f"Error storing conversation: {str(e)}")

    def get_conversation_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Retrieve conversation history"""
        if not self.client:
            return []

        try:
            results = self.conversation_collection.query(
                query_texts=[f"user {user_id}"],
                where={"user_id": user_id},
                n_results=limit
            )

            if results['metadatas'] and len(results['metadatas']) > 0:
                return results['metadatas'][0]
            return []

        except Exception as e:
            print(f"Error retrieving history: {str(e)}")
            return []

    def search_policies(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for relevant policies"""
        if not self.client:
            return []

        try:
            results = self.policy_collection.query(
                query_texts=[query],
                n_results=n_results
            )

            policies = []
            if results['documents'] and results['metadatas']:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    policies.append({"content": doc, "metadata": metadata})

            return policies

        except Exception as e:
            print(f"Error searching policies: {str(e)}")
            return []

    def store_policies_from_csv(self, policies: Dict[str, Any]):
        """Store processed policies in vector database"""
        if not self.client:
            return

        try:
            documents = []
            metadatas = []
            ids = []

            for policy_id, policy_data in policies.items():
                policy_text = f"{policy_data['title']} {policy_data['description']} {policy_data['rules']}"

                documents.append(policy_text)
                metadatas.append({
                    "policy_id": policy_id,
                    "category": policy_data['category'],
                    "title": policy_data['title'],
                    "authorization_level": policy_data['authorization_level']
                })
                ids.append(policy_id)

            if documents:
                self.policy_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

        except Exception as e:
            print(f"Error storing policies: {str(e)}")

@CrewBase
class SupportSystem():
    """SupportSystem crew with alternative web search"""

    def __init__(self):
        # Initialize all components
        self.csv_processor = CSVPolicyProcessor()
        self.vector_store = VectorStoreManager()
        self.web_search_manager = SimpleWebSearchManager()

        # Load CSV policies if available
        self.load_csv_policies()

    def load_csv_policies(self):
        """Load CSV policy files"""
        data_dir = "./data"
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    csv_path = os.path.join(data_dir, file)
                    policies = self.csv_processor.process_csv_file(csv_path)
                    self.vector_store.store_policies_from_csv(policies)
                    print(f"Loaded {len(policies)} policies from {file}")

    # TIER 1: INPUT PROCESSING AGENTS
    @agent
    def ticket_classifier(self) -> Agent:
        return Agent(
            config=self.agents_config['ticket_classifier'],
            verbose=True,
            memory=ShortTermMemory()
        )

    @agent
    def session_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['session_manager'],
            verbose=True,
            memory=LongTermMemory()
        )

    # TIER 2: KNOWLEDGE RETRIEVAL AGENTS
    @agent
    def kb_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['kb_retriever'],
            verbose=True,
            memory=LongTermMemory()
        )

    @agent
    def policy_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['policy_retriever'],
            verbose=True,
            memory=LongTermMemory()
        )

    @agent
    def web_search_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['web_search_coordinator'],
            verbose=True,
            memory=ShortTermMemory()
        )

    # TIER 3: INFORMATION PROCESSING AGENTS
    @agent
    def information_fusion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['information_fusion_agent'],
            verbose=True,
            memory=LongTermMemory()
        )

    @agent
    def qa_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['qa_agent'],
            verbose=True,
            memory=ShortTermMemory()
        )

    # TIER 4: RESPONSE GENERATION AGENTS
    @agent
    def solution_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['solution_generator'],
            verbose=True,
            memory=ShortTermMemory()
        )

    @agent
    def dynamic_responder(self) -> Agent:
        return Agent(
            config=self.agents_config['dynamic_responder'],
            verbose=True,
            memory=ShortTermMemory()
        )

    # TIER 5: SESSION MANAGEMENT AGENTS
    @agent
    def conversation_persister(self) -> Agent:
        return Agent(
            config=self.agents_config['conversation_persister'],
            verbose=True,
            memory=ShortTermMemory()
        )

    # TASK DEFINITIONS
    @task
    def ticket_classification_task(self) -> Task:
        return Task(
            config=self.tasks_config['ticket_classification_task'],
            agent=self.ticket_classifier()
        )

    @task
    def manage_user_session(self) -> Task:
        return Task(
            config=self.tasks_config['manage_user_session'],
            agent=self.session_manager(),
            context=[self.ticket_classification_task()]
        )

    @task
    def kb_retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config['kb_retrieval_task'],
            agent=self.kb_retriever(),
            context=[self.ticket_classification_task()]
        )

    @task
    def retrieve_company_policies(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_company_policies'],
            agent=self.policy_retriever(),
            context=[self.ticket_classification_task()]
        )

    @task
    def coordinate_web_search(self) -> Task:
        return Task(
            config=self.tasks_config['coordinate_web_search'],
            agent=self.web_search_coordinator(),
            context=[self.kb_retrieval_task()]
        )

    @task
    def fuse_multi_source_information(self) -> Task:
        return Task(
            config=self.tasks_config['fuse_multi_source_information'],
            agent=self.information_fusion_agent(),
            context=[self.kb_retrieval_task(), self.retrieve_company_policies(), self.coordinate_web_search()]
        )

    @task
    def solution_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['solution_generation_task'],
            agent=self.solution_generator(),
            context=[self.fuse_multi_source_information()]
        )

    @task
    def create_personalized_response(self) -> Task:
        return Task(
            config=self.tasks_config['create_personalized_response'],
            agent=self.dynamic_responder(),
            context=[self.manage_user_session(), self.solution_generation_task()]
        )

    @task
    def qa_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['qa_review_task'],
            agent=self.qa_agent(),
            context=[self.create_personalized_response()]
        )

    @task
    def persist_interaction_data(self) -> Task:
        return Task(
            config=self.tasks_config['persist_interaction_data'],
            agent=self.conversation_persister(),
            context=[self.qa_review_task()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SupportSystem crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder={
                "provider": "openai",
                "config": {"model": "text-embedding-ada-002"}
            } if os.getenv("OPENAI_API_KEY") else None
        )

    def process_customer_query(self, user_query: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Process customer query with full system capabilities"""

        if not user_id:
            user_id = f"user_{datetime.now().timestamp()}"
        if not session_id:
            session_id = f"session_{datetime.now().timestamp()}"

        # Get conversation history
        conversation_history = self.vector_store.get_conversation_history(user_id)

        # Search internal policies
        relevant_policies = self.vector_store.search_policies(user_query)

        # Prepare inputs
        inputs = {
            "topic": user_query,  # Using 'topic' for backward compatibility
            "user_query": user_query,
            "user_id": user_id,
            "session_id": session_id,
            "user_context": json.dumps({"conversation_history": conversation_history}),
            "internal_policies": json.dumps([p["metadata"] for p in relevant_policies]),
            "current_year": str(datetime.now().year),
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Run the crew
            result = self.crew().kickoff(inputs=inputs)

            # Store the conversation
            self.vector_store.store_conversation(
                user_id=user_id,
                query=user_query,
                response=str(result),
                metadata={"session_id": session_id}
            )

            return {
                "response": result,
                "user_id": user_id,
                "session_id": session_id,
                "sources_used": len(relevant_policies)
            }

        except Exception as e:
            error_response = f"I apologize, but I'm experiencing a technical issue. Please contact our support team directly. Error: {str(e)}"

            return {
                "response": error_response,
                "user_id": user_id,
                "session_id": session_id,
                "error": str(e)
            }
