import os
import json
import os
from google import genai
from google.genai import types
from gtts import gTTS

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
        
        self.sessions_file = "sessions.json"
        self.history = self.load_sessions()
        
        self.load_prompt()

    def load_sessions(self):
        try:
            with open(self.sessions_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_sessions(self):
        """Saves history to sessions.json, ensuring the file doesn't exceed 1MB."""
        max_size = 1 * 1024 * 1024  # 1 MB
        
        # Pruning loop: if the total JSON string size is > 1MB, 
        # remove the oldest entry from the user with the longest history.
        while True:
            json_data = json.dumps(self.history, indent=4)
            if len(json_data) <= max_size:
                break
                
            # Find the user with the most interactions
            best_target = None
            max_len = 0
            for cid, hist in self.history.items():
                if len(hist) > max_len:
                    max_len = len(hist)
                    best_target = cid
            
            if best_target and max_len > 0:
                self.history[best_target].pop(0) # Remove oldest
            else:
                # If all histories are empty but still > 1MB (unlikely but safe), stop
                break

        with open(self.sessions_file, "w") as f:
            f.write(json.dumps(self.history, indent=4))

    def get_history(self, chat_id: str) -> list:
        chat_id_str = str(chat_id)
        if chat_id_str not in self.history:
            self.history[chat_id_str] = []
        return self.history[chat_id_str]

    def add_to_history(self, chat_id: str, user_instruction: str, actions_data: list, final_result: str):
        chat_id_str = str(chat_id)
        hist = self.get_history(chat_id_str)
        hist.append((user_instruction, json.dumps(actions_data), final_result))
        if len(hist) > 100:
            hist.pop(0)
        self.save_sessions()

    def update_last_history_result(self, chat_id: str, refined_result: str):
        chat_id_str = str(chat_id)
        hist = self.get_history(chat_id_str)
        if hist:
            instr, acts, _ = hist[-1]
            hist[-1] = (instr, acts, refined_result)
            self.save_sessions()

    def load_prompt(self):
        try:
            with open("system_prompt.txt", "r") as f:
                self.system_instruction = f.read()
        except FileNotFoundError:
            print("Warning: system_prompt.txt not found. Please ensure it exists.")
            self.system_instruction = ""
            
    async def decide_strategy(self, user_instruction: str, chat_id: int) -> tuple[str, str]:
        """
        Asks Gemini if the user instruction requires a browser or if it can be answered directly.
        Returns a tuple: (strategy, response_text). Strategy can be 'BROWSER' or 'DIRECT'.
        """
        chat_id_str = str(chat_id)
        chat_history = self.get_history(chat_id_str)
        history_context = ""
        if chat_history:
            history_context = "PREVIOUS INTERACTIONS:\n"
            for past_instruction, _, past_result in chat_history:
                history_context += f"- User: {past_instruction}\n  Result: {past_result}\n"

        decision_prompt = f"""
{history_context}
USER REQUEST: {user_instruction}

Analyze the user request. Decide if you need to use a web browser to provide an accurate, up-to-date answer, or if you can answer it directly using your internal knowledge (e.g., general knowledge, math, simple conversation).

Return a JSON object:
{{
  "strategy": "BROWSER" | "DIRECT",
  "reasoning": "short explanation",
  "direct_answer": "Your answer if strategy is DIRECT, otherwise null"
}}
"""
        try:
            response = self.client.models.generate_content(
                model=VISION_MODEL,
                contents=decision_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            data = json.loads(response.text)
            strategy = data.get("strategy", "BROWSER")
            answer = data.get("direct_answer", "")
            return strategy, answer
        except Exception as e:
            print(f"Error deciding strategy: {e}")
            return "BROWSER", ""

    async def refine_answer(self, user_instruction: str, raw_browser_output: str, chat_id: int) -> str:
        """
        Takes the raw output from the autonomous browser loop and asks Gemini to reformat/refine it 
        to perfectly match the user's original request.
        """
        refine_prompt = f"""
USER ORIGINAL REQUEST: {user_instruction}

RAW BROWSER DATA GATHERED:
{raw_browser_output}

Based on the information gathered by the browser agent above, provide a final, polished, and concise answer that directly fulfills the user's request. 
If the technical status says 'Task completed' or 'Action: answer', extract the relevant info and present it professionally.
Do NOT mention the browser actions or JSON technical details. Just answer the user.
"""
        try:
            response = self.client.models.generate_content(
                model=VISION_MODEL,
                contents=refine_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error refining answer: {e}")
            return raw_browser_output

    async def improve_prompt(self):
        try:
            improvement_instruction = '''
Analyze the following system prompt for an AI Web Browsing Agent. 
Please rewrite it to make it more intelligent and robust. 
Specifically, add robust handling for search engine captchas or 'unexpected errors' (e.g., if DuckDuckGo shows an error, instruct the agent to try 'https://lite.duckduckgo.com/lite/' or wait a moment).
CRITICAL: You MUST retain the exact JSON output format requirement and the DuckDuckGo requirement. 
CRITICAL: The new prompt MUST NOT exceed 10000 words.
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
                return new_prompt
            return None
            
        except Exception as e:
            print(f"Error improving prompt: {e}")
            return None

    async def analyze_and_act(self, user_instruction: str, screenshot_bytes: bytes, chat_id: int) -> tuple[str, bool, str]:
        """
        Sends screenshot + instruction to Gemini Vision, gets a JSON action (or array of actions), and executes it.
        Returns a tuple: (status_string, is_done_boolean, audio_path_string) to help the bot know when to stop.
        """
        if not screenshot_bytes:
             return "Error: No browser screenshot available. Did you navigate somewhere?", True, None

        # 1. Compile History into Prompt
        chat_id_str = str(chat_id)
        chat_history = self.get_history(chat_id_str)
        history_context = ""
        if chat_history:
            history_context = "PREVIOUS INTERACTIONS (Use this context to inform your next actions):\n"
            for past_instruction, past_action, past_result in chat_history:
                history_context += f"- User: {past_instruction}\n  Action Taken: {past_action}\n  Result: {past_result}\n"
            history_context += "\n"

        prompt = f"{history_context}CURRENT USER INSTRUCTION: {user_instruction}"
        
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
                actions_data = json.loads(response_text)
            except json.JSONDecodeError:
                return f"Error: Gemini returned invalid JSON: {response_text}", False

            # Ensure we are working with a list of actions
            if isinstance(actions_data, dict):
                actions_data = [actions_data]
            elif not isinstance(actions_data, list):
                return "Error: Gemini did not return a valid action array or object.", False

            total_result_msg = []
            is_done = False
            
            # Execute multiple actions
            for action_data in actions_data:
                action = action_data.get("action")
                reasoning = action_data.get("reasoning", "")
                
                step_msg = f"Action: {action}. {reasoning}"

                if action == "click":
                    coords = action_data.get("coordinates")
                    if coords and len(coords) == 2:
                        await self.browser.click(coords[0], coords[1])
                        step_msg += f" Clicked at {coords}."
                    else:
                        step_msg = "Error: Invalid coordinates for click."

                elif action == "type":
                    text = action_data.get("text")
                    if text:
                        await self.browser.type_text(text)
                        step_msg += f" Typed '{text}'."
                        # Force an Enter press immediately after typing to execute searches
                        await self.browser.press_key("Enter")
                        step_msg += " (Auto-pressed Enter to submit)."

                elif action == "generate_image":
                    text = action_data.get("text")
                    if text:
                        image_result = await self.generate_image(text)
                        step_msg = image_result
                        total_result_msg.append(step_msg)
                        is_done = True
                        break

                elif action == "key":
                    key = action_data.get("key")
                    if key:
                        await self.browser.press_key(key)
                        step_msg += f" Pressed '{key}'."

                elif action == "scroll":
                    direction = action_data.get("direction", "down")
                    await self.browser.scroll(direction)
                    step_msg += f" Scrolled {direction}."

                elif action == "navigate":
                    url = action_data.get("text")
                    if url:
                        await self.browser.navigate(url)
                        step_msg += f" Navigated to {url}."

                elif action == "read":
                    text_content = await self.browser.get_text_content()
                    step_msg += f" Extracted Text: {text_content[:500]}..." # Truncate for logging

                elif action == "answer":
                    step_msg = action_data.get("text", "No answer provided.")
                    total_result_msg.append(step_msg)
                    is_done = True
                    break # Usually the last step

                elif action == "done":
                    step_msg = "Task completed."
                    total_result_msg.append(step_msg)
                    is_done = True
                    break

                total_result_msg.append(step_msg)

            final_result = "\n".join(total_result_msg)
            
            # Save history
            self.add_to_history(chat_id_str, user_instruction, actions_data, final_result)
            
            # Generate Audio TTS if done and NOT an image
            audio_path = None
            if is_done and "IMAGE:" not in final_result:
                try:
                    # Filter out purely technical prefixes for the audio
                    tts_text = final_result.replace("Action: answer. ", "").replace("Extracted Text: ", "")
                    # Limit TTS length if it's crazy long
                    if len(tts_text) > 1000:
                        tts_text = tts_text[:1000] + "... (Text truncated for audio)"
                    
                    tts = gTTS(text=tts_text, lang='en')
                    audio_path = f"tts_{chat_id}.ogg"
                    tts.save(audio_path)
                except Exception as e:
                    print(f"Error generating TTS: {str(e)}")

            return final_result, is_done, audio_path

        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}", False, None

    async def generate_image(self, prompt: str) -> str:
        """
        Uses Gemini Nano Banana to generate an image from a prompt.
        Extracts binary image data from the response and saves it locally.
        """
        try:
            response = self.client.models.generate_content(
                model=IMAGE_GEN_MODEL,
                contents=prompt
            )
            
            # Look for image data in candidates/parts
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if "image" in part.inline_data.mime_type:
                                image_filename = "generated_image.jpg"
                                with open(image_filename, "wb") as f:
                                    f.write(part.inline_data.data)
                                return f"IMAGE:{image_filename}"
            
            # Fallback to response text if no image data found
            if response.text:
                return f"Generated Content: {response.text}"
                
            return "Error: Model returned no image data or text."
            
        except Exception as e:
            return f"Error generating image with {IMAGE_GEN_MODEL}: {str(e)}"

    async def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribes the audio file using Gemini.
        Returns the transcribed text.
        """
        try:
            # Upload the file using the modern genai SDK
            audio_file = self.client.files.upload(file=audio_file_path)
            
            prompt = "Transcribe this audio exactly as it is spoken. Do not include anything else in your response."
            
            response = self.client.models.generate_content(
                model=VISION_MODEL,
                contents=[audio_file, prompt]
            )
            
            # Cleanup uploaded file from Gemini storage
            self.client.files.delete(name=audio_file.name)
            
            return response.text.strip()
            
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"
