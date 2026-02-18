import os
import json
from google import genai
from google.genai import types

# Define your model names here
VISION_MODEL = "gemini-3-flash-preview"
IMAGE_GEN_MODEL = "nano-banana-pro-preview"

class Agent:
    def __init__(self, browser_manager):
        self.browser = browser_manager
        
        try:
            with open("geminiapikey.txt", "r") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise ValueError("geminiapikey.txt file not found. Please create it and add your Gemini API key.")
            
        if not api_key:
            raise ValueError("geminiapikey.txt is empty")
        
        self.client = genai.Client(api_key=api_key)
        
        # System instructions for the vision agent
        self.system_instruction = """
You are a web browsing assistant. You receive screenshots of a web page and user instructions.
You must output a JSON object with the following structure to control the browser:

{
  "action": "click" | "type" | "scroll" | "key" | "navigate" | "done" | "answer",
  "coordinates": [x, y],        // Required for "click"
  "text": "string",             // Required for "type", "navigate", "answer"
  "key": "string",              // Required for "key" (e.g., "Enter")
  "direction": "up" | "down",   // Required for "scroll"
  "reasoning": "brief explanation of why you chose this action"
}

IMPORTANT BROWSER RULES:
- If you need to search for something or open a search engine, you MUST navigate to "https://duckduckgo.com/". 
- Do NOT use Google, Bing, or any other search engine. Always use DuckDuckGo.
- If the instruction is a generic search query and you are not currently on DuckDuckGo, your immediate action should be to navigate to "https://duckduckgo.com/".
- VERY IMPORTANT: After you use the "type" action to enter a search query into DuckDuckGo, your NEXT immediate action in the following turn MUST be a "key" action with "text": "Enter" to actually submit the search.

ACTION GUIDELINES:
- For "click", provide [x, y] coordinates based on the screenshot.
- For "type", provide the text to type into the currently focused field.
- For "navigate", provide the URL.
- For "answer", provide a text response to the user's question based on the page content.
- For "done", indicate the task is complete.
"""

    async def analyze_and_act(self, user_instruction: str, screenshot_bytes: bytes):
        """
        Sends screenshot + instruction to Gemini Vision, gets a JSON action, and executes it.
        Returns a status string or answer to send back to the user.
        """
        if not screenshot_bytes:
             return "Error: No browser screenshot available. Did you navigate somewhere?"

        prompt = f"User Instruction: {user_instruction}"
        
        try:
            response = self.client.models.generate_content(
                model=VISION_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=screenshot_bytes, mime_type="image/jpeg"),
                        ],
                    ),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    temperature=0.4, # Lower temperature for more deterministic actions
                ),
            )
            
            response_text = response.text
            print(f"Gemini Response: {response_text}") # Debug log

            try:
                action_data = json.loads(response_text)
            except json.JSONDecodeError:
                return f"Error: Gemini returned invalid JSON: {response_text}"

            action = action_data.get("action")
            reasoning = action_data.get("reasoning", "")
            
            result_msg = f"Action: {action}. {reasoning}"

            if action == "click":
                coords = action_data.get("coordinates")
                if coords and len(coords) == 2:
                    await self.browser.click(coords[0], coords[1])
                    result_msg += f" Clicked at {coords}."
                else:
                    result_msg = "Error: Invalid coordinates for click."

            elif action == "type":
                text = action_data.get("text")
                if text:
                    await self.browser.type_text(text)
                    result_msg += f" Typed '{text}'."

            elif action == "key":
                key = action_data.get("key")
                if key:
                    await self.browser.press_key(key)
                    result_msg += f" Pressed '{key}'."

            elif action == "scroll":
                direction = action_data.get("direction", "down")
                await self.browser.scroll(direction)
                result_msg += f" Scrolled {direction}."

            elif action == "navigate":
                url = action_data.get("text")
                if url:
                    await self.browser.navigate(url)
                    result_msg += f" Navigated to {url}."

            elif action == "answer":
                return action_data.get("text", "No answer provided.")

            elif action == "done":
                return "Task completed."

            return result_msg

        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}"

    async def generate_image(self, prompt: str) -> str:
        """
        Uses Gemini Nano Banana to generate an image from a prompt.
        In a real scenario, this would call the specific API for image generation.
        For this simplified example, we'll assume the client supports a similar generate_content interface 
        or return a placeholder if not directly supported by this specific SDK version in the same way.
        
        NOTE: 'gemini-nano-banana' is a placeholder name provided by the user. 
        We will attempt to use the model name as requested.
        """
        try:
            # Assuming the imagen/generation API structure. 
            # If the specific 'gemini-nano-banana' uses a different endpoint (e.g. Imagen 3),
            # we would adjust here. For now, we will try standard generation or return a message.
            
            # Since 'gemini-nano-banana' sounds like a custom or future model, 
            # and standard google-genai SDK handles generation mainly for text/multimodal-in, 
            # we'll implement a mock or standard call pattern.
            
            # Actual implementation would depend on the specific API shape for this model.
            # Here we will attempt a text-to-image call if supported, 
            # or return a text description if strictly text-in-text-out.
            
            # For this task, strictly following user request to use that model name.
            # We'll assume it returns a base64 string or url in the response text for this prototype.
            response = self.client.models.generate_content(
                model=IMAGE_GEN_MODEL,
                contents=prompt
            )
            return f"Generated Content: {response.text}" 
        except Exception as e:
            return f"Error generating image with {IMAGE_GEN_MODEL}: {str(e)}"

