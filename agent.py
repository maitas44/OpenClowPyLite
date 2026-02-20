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
        self.learned_optimizations = self.load_learned_optimizations()
        self.system_instruction = ""
        self.load_prompt()

    def load_sessions(self):
        try:
            with open(self.sessions_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_sessions(self):
        """Saves history to sessions.json, ensuring the file doesn't exceed 100MB."""
        max_size = 100 * 1024 * 1024  # 100 MB
        
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

    def load_learned_optimizations(self):
        """Loads learned optimizations from a local file."""
        try:
            if os.path.exists("learned_optimizations.txt"):
                with open("learned_optimizations.txt", "r") as f:
                    return f.read().strip()
        except:
            pass
        return ""

    def add_to_history(self, chat_id: str, user_instruction: str, actions_data: list, final_result: str):
        chat_id_str = str(chat_id)
        hist = self.get_history(chat_id_str)
        # Entry format: [user_instr, actions_json, final_result, verification_status, verification_feedback]
        hist.append([user_instruction, json.dumps(actions_data), final_result, None, None])
        if len(hist) > 1000:
            hist.pop(0)
        self.save_sessions()

    def update_verification_to_history(self, chat_id: str, success: bool, feedback: str):
        chat_id_str = str(chat_id)
        hist = self.get_history(chat_id_str)
        if hist:
            # Update the latest entry (which was just added in solve_autonomous)
            hist[-1][3] = success
            hist[-1][4] = feedback
            self.save_sessions()

    def update_last_history_result(self, chat_id: str, refined_result: str):
        chat_id_str = str(chat_id)
        hist = self.get_history(chat_id_str)
        if hist:
            hist[-1][2] = refined_result
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
LEARNED OPTIMIZATIONS:
{self.learned_optimizations}

USER REQUEST: {user_instruction}

Analyze the user request. Decide if you need to:
1. Use a web BROWSER to provide an accurate, up-to-date answer.
2. Answer DIRECTly using your internal knowledge (for general knowledge, math, conversation).
3. Generate an IMAGE (if the user asks to "draw", "create a picture", "generate an image", etc.).

Return a JSON object:
{{
  "strategy": "BROWSER" | "DIRECT" | "IMAGE",
  "reasoning": "short explanation",
  "direct_answer": "Your answer if strategy is DIRECT, otherwise null",
  "image_prompt": "Specific prompt for the image if strategy is IMAGE, otherwise null"
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
            image_prompt = data.get("image_prompt", user_instruction)
            return strategy, answer, image_prompt
        except Exception as e:
            print(f"Error deciding strategy: {e}")
            return "BROWSER", "", user_instruction

    async def verify_result(self, user_instruction: str, result_text: str = None, image_path: str = None, user_image_path: str = None) -> tuple[bool, str]:
        """
        Asks Gemini to verify if the result (text or image) matches the user's original request.
        Returns a tuple: (is_correct_bool, feedback_message)
        """
        verification_prompt = f"""
USER ORIGINAL REQUEST: {user_instruction}

LEARNED OPTIMIZATIONS:
{self.learned_optimizations}

RESULT TO VERIFY:
{result_text if result_text else "(Image file generated)"}

Did the system successfully fulfill the user's specific request? 
If it was an image request, does the image file exist (indicated by the presence of an image path)?
If it was a text request, is the information accurate and complete?

Return a JSON object:
{{
  "success": true | false,
  "feedback": "Explain why it passed or failed. If it failed, describe what is missing."
}}
"""
        try:
            parts = [types.Part.from_text(text=verification_prompt)]
            
            # Include the generated result image if any
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    img_bytes = f.read()
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
            
            # Include the user's original uploaded image if any
            if user_image_path and os.path.exists(user_image_path):
                with open(user_image_path, "rb") as f:
                    uimg_bytes = f.read()
                parts.append(types.Part.from_bytes(data=uimg_bytes, mime_type="image/jpeg"))

            response = self.client.models.generate_content(
                model=VISION_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            data = json.loads(response.text)
            return data.get("success", False), data.get("feedback", "No feedback provided.")
        except Exception as e:
            print(f"Error during verification: {e}")
            return True, "Verification failed due to technical error, assuming success."

    async def refine_answer(self, user_instruction: str, raw_browser_output: str, chat_id: int) -> str:
        """
        Takes the raw output from the autonomous browser loop and asks Gemini to reformat/refine it 
        to perfectly match the user's original request.
        """
        refine_prompt = f"""
USER ORIGINAL REQUEST: {user_instruction}

LEARNED OPTIMIZATIONS:
{self.learned_optimizations}

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
            # Load session history for analysis
            history_summary = ""
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, "r") as f:
                    sessions = json.load(f)
                    # Extract recent failures or noteworthy interactions
                    count = 0
                    for cid, hist in sessions.items():
                        for entry in hist[-10:]: # Look at last 10 per user
                            if len(entry) >= 5 and entry[3] is False: # It failed
                                history_summary += f"FAILED TASK: {entry[0]}\nREASON: {entry[4]}\n"
                                count += 1
                                if count > 20: break
            
            improvement_instruction = f'''
Analyze the following session history and the current system prompt.
Your goal is to evolve the bot's behavior to avoid these failures in the future.

HISTORICAL FAILURES:
{history_summary}

CURRENT SYSTEM PROMPT:
{self.system_instruction}

CURRENT LEARNED OPTIMIZATIONS:
{self.learned_optimizations}

TASK:
1. Provide a REWRITTEN system prompt for `system_prompt.txt`.
2. Provide a set of "LEARNED OPTIMIZATIONS" (brief rules) that should be applied to the bot's internal logic (Decision, Verification, Refinement).

Return a JSON object:
{{
  "new_system_prompt": "the full text",
  "new_learned_optimizations": "brief bullet points of learned rules"
}}
'''
            try:
                response = self.client.models.generate_content(
                    model=VISION_MODEL,
                    contents=improvement_instruction,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                data = json.loads(response.text)
                new_prompt = data.get("new_system_prompt")
                new_opts = data.get("new_learned_optimizations")

                if new_prompt:
                    self.system_instruction = new_prompt
                    with open("system_prompt.txt", "w") as f:
                        f.write(new_prompt)
                
                if new_opts:
                    self.learned_optimizations = new_opts
                    with open("learned_optimizations.txt", "w") as f:
                        f.write(new_opts)

                return "Prompts improved based on session history!"
            except Exception as inner:
                print(f"Inner improvement error: {inner}")
                return None
        
        except Exception as e:
            print(f"Error in improve_prompt: {e}")
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
                
                print(f"Executing Action: {action} ({reasoning})")
                
                if action == "navigate":
                    await self.browser.navigate(text)
                elif action == "click":
                    await self.browser.click(text)
                elif action == "type":
                    # text field might be "selector|text"
                    if "|" in text:
                        sel, val = text.split("|", 1)
                        await self.browser.type_text(sel, val)
                elif action == "scroll":
                    direction = text.lower() if text else "down"
                    await self.browser.scroll(direction)
                elif action == "wait":
                    await asyncio.sleep(2)
                elif action == "generate_image":
                    final_result = await self.generate_image(text)
                    is_done = True
                    break
                elif action == "answer":
                    final_result = text
                    is_done = True
                    break
                elif action == "done":
                    final_result = text or "Task completed."
                    is_done = True
                    break
            
            # Save to history
            self.add_to_history(chat_id_str, user_instruction, actions, final_result)
            
            # Generate TTS for the final answer if done
            tts_path = None
            if is_done and final_result and not final_result.startswith("IMAGE:"):
                try:
                    # Clean up the text for gTTS (remove special markdown characters)
                    clean_text = final_result.replace("**", "").replace("_", "").replace("`", "")
                    tts = gTTS(text=clean_text[:1000], lang='en')
                    tts_path = f"tts_{chat_id}.voice"
                    tts.save(tts_path)
                except Exception as tts_e:
                    print(f"TTS Error: {tts_e}")

            return final_result or "Processing...", is_done, tts_path

        except Exception as e:
            print(f"Error in analyze_and_act: {e}")
            return f"Error: {e}", True, None

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
