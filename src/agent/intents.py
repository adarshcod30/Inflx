from pydantic import BaseModel, Field
from typing import Optional

class IntentClassification(BaseModel):
    """Classification of the user's intent."""
    intent: str = Field(description="The classified intent: 'greeting', 'product_inquiry', or 'high_intent_lead'")
    reasoning: str = Field(description="Brief explanation for the classification")

class LeadDetails(BaseModel):
    """Extracted lead details from conversation."""
    name: Optional[str] = Field(None, description="The user's full name")
    email: Optional[str] = Field(None, description="The user's email address")
    platform: Optional[str] = Field(None, description="The creator platform (YouTube, Instagram, etc.)")
