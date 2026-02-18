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
        
        self.load_prompt()

    def load_prompt(self):
        try:
            with open("system_prompt.txt", "r") as f:
                self.system_instruction = f.read()
        except FileNotFoundError:
            print("Warning: system_prompt.txt not found. Please ensure it exists.")
            self.system_instruction = ""
            
    async def improve_prompt(self):
        try:
            improvement_instruction = '''
Analyze the following system prompt for an AI Web Browsing Agent. 
Please rewrite it to make it more intelligent and robust. 
Specifically, add robust handling for search engine captchas or 'unexpected errors' (e.g., if DuckDuckGo shows an error, instruct the agent to try 'https://html.duckduckgo.com/html/' or wait a moment).
CRITICAL: You MUST retain the exact JSON output format requirement and the DuckDuckGo requirement. 
Respond ONLY with the completely rewritten prompt text, with no markdown code blocks wrapping it.
'''
            
            prompt = f"{improvement_instruction}\n\nCURRENT PROMPT:\n{self.system_instruction}"
            
            response = self.client.models.generate_content(
                model=VISION_MODEL,
                contents=prompt,
            )
            
            new_prompt = response.text.strip()
            
            # Basic validation to ensure it didn't just return garbage
            if "{" in new_prompt and "action" in new_prompt:
                self.system_instruction = new_prompt
                with open("system_prompt.txt", "w") as f:
                    f.write(new_prompt)
                return True
            return False
            
        except Exception as e:
            print(f"Error improving prompt: {e}")
            return False

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
                    # Force an Enter press immediately after typing to execute searches
                    await self.browser.press_key("Enter")
                    result_msg += " (Auto-pressed Enter to submit)."

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

