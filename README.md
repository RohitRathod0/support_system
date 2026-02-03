# SupportSystem Crew

🚀 Support System — AI-Powered Multi-Agent Support Platform

This project implements a customer support system built with a multi-agent AI architecture using crewAI and Retrieval-Augmented Generation (RAG) concepts. The system allows multiple intelligent agents to work together on complex tasks — such as generating reports, answering questions, or processing inputs — using LLMs and vector databases.

Built as a flexible AI support backbone with modular configuration and agent logic.

🔍 About the Project

Support System is a collaborative AI system where multiple agents, each with their own capabilities and tasks, work together to achieve complex goals. This system is highly customizable — define your agents and tasks using YAML configuration, inject your logic, and execution happens through a customizable pipeline.

Powered by:

🧠 crewAI for agent orchestration

📚 RAG (Retrieval-Augmented Generation)

💡 Config-driven workflows

🧱 Architecture Overview

Here’s how Support System operates:

Agents Configuration

Defined in config/agents.yaml.

Each agent has specific abilities, prompts, and tools.

Tasks Definition

Stored in config/tasks.yaml.

Specifies tasks for each agent to complete.

Crew Runner

crewai run initializes all agents and task assignments.

Vector DB + RAG (optional)

Supports knowledge retrieval for agents using vector embeddings.

Output

The system executes and outputs results (e.g., report files) based on your task definitions.


🛠️ Tech Stack
Layer	Technology
Language:	Python
Agent Framework	crewAI
LLM Integration	Ollama, Custom Models
Vector Database	Vector DB (folder: vector_db)
Dependency Management	uv (like poetry/pipenv)
Config Files	YAML
Env Management	.env
RAG Support	Yes
	
🚀 Installation

📌 Requirements

Python 3.10 — Python 3.13

uv for package management

# Install uv if you haven’t already
pip install uv


After cloning:

git clone https://github.com/RohitRathod0/support_system.git
cd support_system

# Install dependencies
uv install

▶️ How to Run the Project

Before running, set your OPENAI_API_KEY (or equivalent API key) in a .env file:

OPENAI_API_KEY=your_api_key_here


You can then start the agents:

crewai run


This will initialize the support crew and execute all defined tasks. By default, one of the example outputs is a generated report.md in the project root.

🧩 Configuration
🛠 Agents

Edit the core agent definitions in:

src/support_system/config/agents.yaml


Each agent can have:

Name

Goals

Tools

Prompts


🗂 Tasks

The tasks you want agents to perform are configured in:

src/support_system/config/tasks.yaml


Example tasks can include:

Research tasks

Data extraction

Text creation

Q&A chains


🏗️ Core Components
Component	Purpose
main.py	Entry point for custom input logic
crew.py	Agent orchestration & agent setup
agents.yaml	Agent definitions
tasks.yaml	Task assignments
vector_db/	Local vector store for RAG searches
data/knowledge	Static data to augment agents
config/	All YAML configuration files
	
📁 Project Structure
support_system/
│
├── config/
│   ├── agents.yaml
│   └── tasks.yaml
│
├── data/
│   └── knowledge/
├── src/
│   └── support_system/
│       ├── crew.py
│       ├── main.py
│       └── ...
├── vector_db/
├── .env
├── requirements.txt
└── README.md


🤝 Contribution

Contributions are welcome! 🚀

Whether it’s:

More agent logic

Better RAG integration

New workflows

UI front-end

Feel free to open issues or submit pull requests.

📬 Support & Contact

If you need help or have questions:

📄 crewAI documentation: https://docs.crewai.com

💬 Join the crewAI Discord

🤝 Open an issue in this repository
🧠 What I Learned From This Project

Designing multi-agent AI systems instead of single-agent pipelines

Writing config-driven architectures using YAML

Understanding how agents collaborate and delegate tasks

Integrating LLMs with retrieval systems (RAG)

Structuring large AI projects for scalability

Managing environment variables and secrets securely

Debugging agent hallucinations and prompt failures

Using vector databases for semantic search

Designing modular AI workflows

This project strengthened my understanding of agentic AI design patterns and production-style project structuring.

⚙️ Development Process

Researched multi-agent frameworks

Chose crewAI for orchestration

Designed folder structure

Created agent and task schemas

Implemented orchestration logic

Integrated vector database

Tested agent collaboration

Tuned prompts and tasks

Added output generation

🔁 System Workflow (Step-by-Step)

User provides input

Crew initializes agents

Tasks loaded from YAML

Agents receive instructions

Agents fetch knowledge (RAG)

LLM generates reasoning

Agents collaborate

Results aggregated

Final output produced
