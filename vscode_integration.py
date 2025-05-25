"""
VS Code server integration utilities for Devstral Modal deployment.
"""

import json
import os
from typing import Dict, Any, Optional
import httpx

class DevstralVSCodeClient:
    """Client for integrating Devstral with VS Code servers."""
    
    def __init__(self, modal_endpoint: str, api_key: Optional[str] = None):
        self.base_url = modal_endpoint.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def code_completion(
        self, 
        code: str, 
        language: str = "python",
        context: str = "",
        max_tokens: int = 1024
    ) -> str:
        """Get code completion from Devstral."""
        
        prompt = f"""Complete the following {language} code:

Context: {context}

Code to complete:
```{language}
{code}