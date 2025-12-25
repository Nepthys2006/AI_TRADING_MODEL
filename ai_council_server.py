"""
AI Trading Council Server
=========================
A FastAPI server that orchestrates multiple Ollama AI models as a council of
senior forex trading experts. They collaborate, share ideas, and rank each other's responses.
"""

import asyncio
import json
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, List, Dict
from datetime import datetime
import random

app = FastAPI(title="AI Trading Council")


# ======== In-Memory Conversation Database ========
class ConversationDatabase:
    """
    In-memory database for storing conversation history.
    Automatically resets when the server is stopped/restarted.
    """
    def __init__(self, max_history: int = 50):
        self.conversations: List[Dict] = []
        self.max_history = max_history  # Limit to prevent memory issues
    
    def add_conversation(self, question: str, responses: List[Dict], synthesis: str, rankings: List[Dict] = None):
        """Add a new conversation to the database"""
        entry = {
            "id": len(self.conversations) + 1,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "responses": [
                {
                    "model_name": r.get("model_name", "Unknown"),
                    "specialty": r.get("specialty", "Unknown"),
                    "response": r.get("response", "")[:500]  # Truncate for memory efficiency
                }
                for r in responses
            ],
            "synthesis": synthesis[:1000] if synthesis else "",  # Truncate synthesis
            "rankings": rankings or []
        }
        self.conversations.append(entry)
        
        # Keep only the most recent conversations to prevent memory issues
        if len(self.conversations) > self.max_history:
            self.conversations = self.conversations[-self.max_history:]
    
    def get_context_summary(self, max_entries: int = 5) -> str:
        """Get a detailed summary of recent conversations for context"""
        if not self.conversations:
            return ""
        
        recent = self.conversations[-max_entries:]
        context_parts = ["=== PREVIOUS COUNCIL DISCUSSIONS (REMEMBER THESE) ===\n"]
        
        for conv in recent:
            context_parts.append(f"\n[Session #{conv['id']} - {conv['timestamp'][:16]}]")
            context_parts.append(f"USER QUESTION: {conv['question']}")
            
            # Include key responses from each model
            context_parts.append("\nEXPERT RESPONSES:")
            for resp in conv.get('responses', []):
                context_parts.append(f"  - {resp['model_name']} ({resp['specialty']}): {resp['response'][:200]}...")
            
            # Include rankings
            if conv.get('rankings'):
                context_parts.append("\nRANKINGS:")
                for rank in conv['rankings'][:3]:  # Top 3
                    context_parts.append(f"  #{rank.get('votes', 0)}: {rank.get('model_name', 'Unknown')} - Score: {rank.get('score', 0)}/10")
            
            # Include synthesis
            context_parts.append(f"\nCOUNCIL SYNTHESIS: {conv['synthesis']}")
            context_parts.append("\n" + "-"*50)
        
        context_parts.append("\n=== END OF PREVIOUS DISCUSSIONS ===")
        context_parts.append("\nUse this history to provide continuity. If the user references a previous discussion, build upon it.")
        
        return "\n".join(context_parts)
    
    def get_all_conversations(self) -> List[Dict]:
        """Return all stored conversations"""
        return self.conversations
    
    def get_conversation_count(self) -> int:
        """Return the number of stored conversations"""
        return len(self.conversations)
    
    def clear_history(self):
        """Clear all conversation history"""
        self.conversations = []


# Initialize the global conversation database
conversation_db = ConversationDatabase()

# Ollama API endpoint
OLLAMA_BASE_URL = "http://localhost:11434"

# AI Council Members - Each model with a persona
AI_COUNCIL = [
    {"id": "nemotron-3-nano:30b-cloud", "name": "Nemotron", "color": "#FF6B6B", "specialty": "Technical Analysis"},
    {"id": "deepseek-v3.2:cloud", "name": "DeepSeek V3.2", "color": "#4ECDC4", "specialty": "Market Sentiment"},
    {"id": "minimax-m2:cloud", "name": "MiniMax M2", "color": "#45B7D1", "specialty": "Risk Management"},
    {"id": "gemini-3-pro-preview:latest", "name": "Gemini Pro", "color": "#96CEB4", "specialty": "Macro Economics"},
    {"id": "kimi-k2:1t-cloud", "name": "Kimi K2", "color": "#FFEAA7", "specialty": "Price Action"},
    {"id": "glm-4.6:cloud", "name": "GLM 4.6", "color": "#DDA0DD", "specialty": "Quantitative Analysis"},
    {"id": "qwen3-vl:235b-cloud", "name": "Qwen3 VL", "color": "#98D8C8", "specialty": "Pattern Recognition"},
    {"id": "deepseek-v3.1:671b-cloud", "name": "DeepSeek V3.1", "color": "#F7DC6F", "specialty": "Trend Following"},
    {"id": "gpt-oss:120b-cloud", "name": "GPT-OSS 120B", "color": "#BB8FCE", "specialty": "Fundamental Analysis"},
    {"id": "gpt-oss:20b-cloud", "name": "GPT-OSS 20B", "color": "#85C1E9", "specialty": "Swing Trading"},
    {"id": "qwen3-coder:480b-cloud", "name": "Qwen3 Coder", "color": "#F1948A", "specialty": "Algorithmic Strategies"},
]

# System prompt for senior forex expert persona
SENIOR_EXPERT_PROMPT = """You are a highly respected senior forex trading expert with over 20 years of experience in the global currency markets. You have:
- Managed multi-million dollar portfolios for major investment banks
- Developed proprietary trading strategies that have consistently outperformed the market
- Mentored hundreds of junior traders who have gone on to successful careers
- Published research papers on market microstructure and algorithmic trading

Your specialty is: {specialty}

You are part of an elite AI Trading Council where the world's best trading minds collaborate. You take pride in your expertise and always aim to provide the most valuable insights.

{conversation_context}

When responding:
1. Be confident but humble - acknowledge when others have valid points
2. Share specific, actionable insights based on your expertise
3. Reference technical concepts and real-world trading scenarios
4. Keep responses focused and professional (3-5 paragraphs max)
5. Your specialty is {specialty}, so emphasize insights from that perspective
6. If the user references a previous question, use the conversation history above to provide context-aware responses

Respond as a senior expert would - with authority and depth."""

# Ranking prompt for AIs to evaluate other responses
RANKING_PROMPT = """You are a senior forex trading expert evaluating responses from your colleagues on the AI Trading Council.

The original question was:
"{question}"

Here are the responses from other council members:
{responses}

Rate EACH response on a scale of 1-10 based on:
- Accuracy of forex/trading knowledge (30%)
- Actionability of advice (25%)
- Depth of insight (25%)
- Clarity of explanation (20%)

Respond in this EXACT JSON format only (no other text):
{{
  "rankings": [
    {{"model_name": "NAME", "score": X, "reason": "brief reason"}}
  ],
  "best_insight": "The most valuable insight from the discussion"
}}"""

# Collaboration synthesis prompt
COLLABORATION_PROMPT = """You are the senior moderator of an elite AI Trading Council. Your job is to synthesize the best ideas from all council members into a cohesive trading strategy recommendation.

The original question was:
"{question}"

Here are all the expert responses:
{responses}

Here are the rankings from the council:
{rankings}

Create a synthesis that:
1. Identifies the consensus view (if any)
2. Highlights the most valuable unique insights
3. Provides a clear, actionable recommendation
4. Notes any important disagreements or risks

Keep your synthesis to 4-6 paragraphs. Be authoritative and professional."""


class ConnectionManager:
    """Manages WebSocket connections"""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


async def check_model_status(model_id: str) -> bool:
    """Check if a specific Ollama model is available"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name") == model_id for m in models)
    except Exception:
        pass
    return False


async def get_all_models_status() -> list[dict]:
    """Get status of all council members"""
    statuses = []
    for ai in AI_COUNCIL:
        is_online = await check_model_status(ai["id"])
        statuses.append({
            **ai,
            "online": is_online
        })
    return statuses


async def query_ollama(model_id: str, prompt: str, system: str = "") -> Optional[str]:
    """Query a specific Ollama model"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": model_id,
                "prompt": prompt,
                "system": system,
                "stream": False
            }
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")
    except Exception as e:
        print(f"Error querying {model_id}: {e}")
    return None


async def get_council_response(model: dict, question: str) -> dict:
    """Get response from a single council member"""
    # Get conversation context from previous sessions
    context = conversation_db.get_context_summary()
    context_text = context if context else "This is the first question in this session."
    
    system_prompt = SENIOR_EXPERT_PROMPT.format(
        specialty=model["specialty"],
        conversation_context=context_text
    )
    response = await query_ollama(model["id"], question, system_prompt)
    
    return {
        "model_id": model["id"],
        "model_name": model["name"],
        "color": model["color"],
        "specialty": model["specialty"],
        "response": response if response else "Unable to generate response at this time.",
        "success": response is not None
    }


async def get_rankings(ranker_model: dict, question: str, responses: list[dict]) -> Optional[dict]:
    """Have one AI rank the other responses"""
    # Format responses for the ranking prompt
    responses_text = "\n\n".join([
        f"**{r['model_name']}** ({r['specialty']}):\n{r['response']}"
        for r in responses if r["model_id"] != ranker_model["id"]
    ])
    
    prompt = RANKING_PROMPT.format(question=question, responses=responses_text)
    
    # Get conversation context and format system prompt properly
    context = conversation_db.get_context_summary()
    context_text = context if context else "This is the first question in this session."
    system = SENIOR_EXPERT_PROMPT.format(
        specialty=ranker_model["specialty"],
        conversation_context=context_text
    )
    
    result = await query_ollama(ranker_model["id"], prompt, system)
    
    if result:
        try:
            # Try to extract JSON from the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(result[json_start:json_end])
        except json.JSONDecodeError:
            print(f"Failed to parse ranking JSON from {ranker_model['name']}: {result[:200]}")
    return None


async def synthesize_collaboration(question: str, responses: list[dict], rankings: list[dict]) -> str:
    """Synthesize all responses into a collaborative recommendation"""
    responses_text = "\n\n".join([
        f"**{r['model_name']}** ({r['specialty']}):\n{r['response']}"
        for r in responses
    ])
    
    rankings_text = "\n".join([
        f"- {r.get('ranker', 'Unknown')}: Best insight - {r.get('best_insight', 'N/A')}"
        for r in rankings if r
    ])
    
    prompt = COLLABORATION_PROMPT.format(
        question=question,
        responses=responses_text,
        rankings=rankings_text
    )
    
    # Use a random available model as moderator
    available_models = [m for m in AI_COUNCIL]
    moderator = random.choice(available_models)
    
    system = "You are the senior moderator synthesizing insights from the AI Trading Council."
    result = await query_ollama(moderator["id"], prompt, system)
    
    return result if result else "Unable to synthesize collaboration at this time."


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Send initial model statuses
    statuses = await get_all_models_status()
    await websocket.send_json({
        "type": "model_status",
        "data": statuses
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "start_council":
                question = data.get("question", "")
                
                if not question:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Please provide a question for the council."
                    })
                    continue
                
                await websocket.send_json({
                    "type": "council_started",
                    "message": f"Council convened to discuss: {question}"
                })
                
                # Phase 1: Gather responses from all council members
                responses = []
                for model in AI_COUNCIL:
                    await websocket.send_json({
                        "type": "model_thinking",
                        "model_id": model["id"],
                        "model_name": model["name"]
                    })
                    
                    response = await get_council_response(model, question)
                    responses.append(response)
                    
                    await websocket.send_json({
                        "type": "model_response",
                        "data": response
                    })
                
                # Phase 2: Have 3 random models rank the responses
                await websocket.send_json({
                    "type": "ranking_started",
                    "message": "Council members are now evaluating each other's responses..."
                })
                
                rankers = random.sample(AI_COUNCIL, min(3, len(AI_COUNCIL)))
                all_rankings = []
                aggregated_scores = {r["model_name"]: [] for r in responses}
                
                for ranker in rankers:
                    await websocket.send_json({
                        "type": "ranker_thinking",
                        "ranker_name": ranker["name"]
                    })
                    
                    ranking_result = await get_rankings(ranker, question, responses)
                    
                    if ranking_result:
                        ranking_result["ranker"] = ranker["name"]
                        all_rankings.append(ranking_result)
                        
                        # Aggregate scores
                        for r in ranking_result.get("rankings", []):
                            model_name = r.get("model_name")
                            score = r.get("score", 5)
                            if model_name in aggregated_scores:
                                aggregated_scores[model_name].append(score)
                        
                        await websocket.send_json({
                            "type": "ranking_result",
                            "ranker": ranker["name"],
                            "data": ranking_result
                        })
                
                # Calculate final scores
                final_rankings = []
                for model_name, scores in aggregated_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        final_rankings.append({
                            "model_name": model_name,
                            "score": round(avg_score, 1),
                            "votes": len(scores)
                        })
                
                final_rankings.sort(key=lambda x: x["score"], reverse=True)
                
                await websocket.send_json({
                    "type": "final_rankings",
                    "data": final_rankings
                })
                
                # Phase 3: Synthesize collaboration
                await websocket.send_json({
                    "type": "synthesis_started",
                    "message": "Synthesizing collaborative insights..."
                })
                
                synthesis = await synthesize_collaboration(question, responses, all_rankings)
                
                await websocket.send_json({
                    "type": "synthesis_complete",
                    "data": synthesis
                })
                
                # Save conversation to in-memory database
                conversation_db.add_conversation(
                    question=question,
                    responses=responses,
                    synthesis=synthesis,
                    rankings=final_rankings
                )
                
                await websocket.send_json({
                    "type": "council_complete",
                    "message": f"Council session complete! (Session #{conversation_db.get_conversation_count()})"
                })
            
            elif data.get("action") == "refresh_status":
                statuses = await get_all_models_status()
                await websocket.send_json({
                    "type": "model_status",
                    "data": statuses
                })
            
            elif data.get("action") == "clear_history":
                conversation_db.clear_history()
                await websocket.send_json({
                    "type": "history_cleared",
                    "message": "Conversation history has been cleared."
                })
            
            elif data.get("action") == "get_history":
                await websocket.send_json({
                    "type": "history_data",
                    "data": conversation_db.get_all_conversations(),
                    "count": conversation_db.get_conversation_count()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ======== REST API Endpoints for Conversation History ========
@app.get("/api/history")
async def get_history():
    """Get all conversation history"""
    return {
        "conversations": conversation_db.get_all_conversations(),
        "count": conversation_db.get_conversation_count()
    }


@app.get("/api/history/count")
async def get_history_count():
    """Get the number of stored conversations"""
    return {"count": conversation_db.get_conversation_count()}


@app.delete("/api/history/clear")
async def clear_history():
    """Clear all conversation history"""
    conversation_db.clear_history()
    return {"message": "Conversation history cleared", "count": 0}


# Mount static files and serve index
app.mount("/static", StaticFiles(directory="council_ui"), name="static")


@app.get("/")
async def root():
    return FileResponse("council_ui/index.html")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  AI TRADING COUNCIL SERVER")
    print("  In-Memory Conversation History: ENABLED")
    print("="*60)
    print("\n  Starting server at http://localhost:8000")
    print("  Make sure Ollama is running with your models!")
    print("\n  Note: Conversation history will reset when server stops.\n")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
