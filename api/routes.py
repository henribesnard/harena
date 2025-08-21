from fastapi import APIRouter

from models.conversation_models import AgentQueryRequest, AgentQueryResponse
from teams.team_orchestrator import TeamOrchestrator

router = APIRouter(tags=["conversation"])
orchestrator = TeamOrchestrator()


@router.post("/chat", response_model=AgentQueryResponse)
async def chat_endpoint(payload: AgentQueryRequest) -> AgentQueryResponse:
    """Proxy a chat message through the team orchestrator."""
    conv_id = orchestrator.start_conversation()
    reply = await orchestrator.query_agents(conv_id, payload.message)
    return AgentQueryResponse(conversation_id=conv_id, reply=reply)
