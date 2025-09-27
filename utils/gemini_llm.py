import os
import aiohttp
import json
import backoff
from mcp_agent.workflows.llm.augmented_llm import RequestParams

class GeminiAugmentedLLM:
    """
    A custom LLM class to interact directly with the Google Gemini API's `generateContent` endpoint.
    This class is designed to be a replacement for the OpenAI-compatible approach when it's not available.
    """

    def __init__(self, agent, **kwargs):
        """
        Initializes the GeminiAugmentedLLM.

        Args:
            agent: The parent agent instance.
            **kwargs: Additional keyword arguments.
        """
        self.agent = agent
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent?key={self.api_key}"

    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=5)
    async def generate_str(self, message: str, request_params: RequestParams = None) -> str:
        """
        Generates content using the native Gemini API.

        Args:
            message: The prompt message to send to the model.
            request_params: An object containing request parameters like max_tokens and temperature.

        Returns:
            The generated text content from the Gemini API.
        """
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": message}
                    ]
                }
            ]
        }

        generation_config = {}
        if request_params:
            if hasattr(request_params, 'max_tokens') and request_params.max_tokens is not None:
                generation_config['maxOutputTokens'] = request_params.max_tokens
            if hasattr(request_params, 'temperature') and request_params.temperature is not None:
                generation_config['temperature'] = request_params.temperature

        if generation_config:
            payload['generationConfig'] = generation_config

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    # Check for candidates and content in the response
                    if 'candidates' in data and data['candidates']:
                        candidate = data['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                            return candidate['content']['parts'][0].get('text', '')
                    # If the expected structure isn't there, return the raw response for debugging
                    return json.dumps(data)
                else:
                    error_text = await response.text()
                    print(f"Error from Gemini API: {response.status} {error_text}")
                    # Re-raise the error to be caught by backoff
                    response.raise_for_status()
                    return f"Error: Failed to get response from Gemini API. Status: {response.status}"