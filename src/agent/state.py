from typing import TypedDict, Annotated, Sequence, List, Optional
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # Chat history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Intent tracking
    intent: Optional[str]
    
    # Lead details (Slots)
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    
    # Metadata
    is_tool_called: bool
