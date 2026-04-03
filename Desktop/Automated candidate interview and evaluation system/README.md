# Automated Candidate Interview & Evaluation System (ARIES)

ARIES is an agentic AI solution for top-of-funnel recruitment.
It conducts technical interview rounds, evaluates candidate responses in real time, and produces a final recommendation.

This implementation follows a multi-agent architecture inspired by your transcript:

- Interviewer Agent: generates interview questions
- Candidate Bridge: captures candidate input
- Evaluation Agent: scores and critiques answers
- FastAPI + WebSocket: streams the interview flow in real time
- Microsoft AutoGen (AgentChat): orchestration framework

## System Architecture

Flow per round:

1. Interviewer Agent asks a question
2. Candidate submits an answer
3. Evaluation Agent scores and provides feedback
4. Interviewer Agent asks next question using prior context

On stop, ARIES produces a final recommendation report (Strengths, Gaps, Hire/No Hire).

## Section-Wise Build Path

Section 1: Foundations & Configuration

1. Project setup (venv, dependencies)
2. AutoGen + model configuration
3. Environment variables

Section 2: Agent Implementation

1. Interviewer Agent
2. Candidate proxy/bridge
3. Evaluation Agent

Section 3: Integration & Testing

1. WebSocket integration of all agents
2. Multi-round interview testing

Section 4: Interface & Deployment

1. Interactive browser interface
2. Render deployment-ready structure

## Quick Start

1. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
cp .env.example .env
```

4. Start Ollama (free local model API):

```bash
ollama serve
ollama pull llama3.2:3b
```

5. Run server:

```bash
uvicorn app.main:app --reload
```

6. Open UI:

http://127.0.0.1:8000

## API Endpoints

- GET /: Interview UI
- GET /health: Service health check
- WS /ws/interview: Real-time interview channel

WebSocket actions:

- start: begin interview with topic
- answer: submit candidate answer
- stop: end interview and receive final report

## Project Structure

app/

- core/config.py: environment settings
- core/autogen_client.py: model client for Ollama OpenAI-compatible API
- agents/workflow.py: interviewer/evaluator orchestration
- main.py: FastAPI routes + WebSocket handling
- static/index.html: frontend interview console

## Notes

- This project uses free models through Ollama.
- You can replace Ollama with any OpenAI-compatible endpoint later.
