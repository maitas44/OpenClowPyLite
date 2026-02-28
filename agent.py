import os
import json
import asyncio
from google import genai
from google.genai import types
from gtts import gTTS
from planner import Planner
from memory import Memory

# Ranking of model families (higher is smarter)
# Note: models will be searched as substrings
MODEL_FAMILY_PRIORITY = [
    "gemini-3.1-pro",
    "gemini-3-pro",
    "gemini-2.5-pro",
    "gemini-1.5-pro",
    "gemini-pro",
    "gemini-3.1-flash",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-flash",
    "gemini-flash-lite",
    "gemma"
]

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
        
        # Set environment variable as fallback for some SDK calls
        os.environ["GOOGLE_API_KEY"] = api_key
        
        self.client = genai.Client(api_key=api_key)
        
        # Rank available models by smartness
        self.ranked_models = self._get_ranked_models()
        self.image_models = self._rank_image_models()
        print(f"[MODELS] Found {len(self.ranked_models)} text models, {len(self.image_models)} image models.")
        
        self.planner = Planner(self)
        self.memory = Memory()
        self.current_plan = None

        self.sessions_file = "sessions.json"
        self.history = self.load_sessions()
        self.learned_optimizations = self.load_learned_optimizations()
        self.system_instruction = ""
        self.load_prompt()
        
        # Per-task step journal: reset at the start of each browser task.
        self._task_steps: list = []
        self._task_screenshots: list = [] # Rolling buffer of last 10 screenshots
        
        # Legacy fingerprint fields (kept for compat, no longer used for detection)
        self._last_action_fingerprint: str = ""
        self._stuck_count: int = 0

    def _get_ranked_models(self):
        """Discovers and ranks available Gemini models for the current API key."""
        try:
            available = list(self.client.models.list())
            usable_models = []
            for m in available:
                # We want models that support text/image generation
                # In the new genai SDK, we check model names for keywords if attributes are missing
                name = m.name.lower()
                if "embedding" in name or "aqa" in name or "imagen" in name or "veo" in name:
                    continue
                
                # Assign a score based on family priority
                score = -1
                for i, family in enumerate(MODEL_FAMILY_PRIORITY):
                    if family in name:
                        score = len(MODEL_FAMILY_PRIORITY) - i
                        break
                
                if score >= 0:
                    usable_models.append((m.name, score))
            
            # Sort by score (descending)
            usable_models.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in usable_models]
        except Exception as e:
            # If listing fails with 400 (API Key not found), it might be an SDK quirk.
            # We already set the env var above, so subsequent calls might still work.
            print(f"[MODELS] Warning: Could not list models ({e}). Models may still work if the key is valid.")
            # Fallback to a common list of models
            return ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.0-pro"]

    def _rank_image_models(self):
        """Finds and ranks available image generation models."""
        try:
            available = list(self.client.models.list())
            image_models = []
            priority = ["imagen-4.0-ultra", "imagen-4.0", "nano-banana-pro", "nano-banana", "gemini-2.0-flash-exp-image-generation"]
            for m in available:
                name = m.name.lower()
                if "generateContent" in m.supported_generation_methods or "generate_image" in name: # fallback check
                    score = -1
                    for i, p in enumerate(priority):
                        if p in name:
                            score = len(priority) - i
                            break
                    if score >= 0:
                        image_models.append((m.name, score))
            
            image_models.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in image_models]
        except:
            return ["models/nano-banana-pro-preview"]

    async def _call_gemini(self, contents, config, candidates=None):
        """Attempts to call Gemini using a list of models, falling back on failure."""
        models_to_try = candidates if candidates else self.ranked_models
        last_error = None
        
        for model_name in models_to_try:
            try:
                print(f"[FALLBACK] Trying model: {model_name}")
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                return response
            except Exception as e:
                err_str = str(e).lower()
                # Check if it's a retryable error (quota, rate limit, internal)
                if any(x in err_str for x in ["429", "resource_exhausted", "quota", "exhausted", "500", "internal"]):
                    print(f"[FALLBACK] Model {model_name} failed: {e}. Trying next...")
                    last_error = e
                    await asyncio.sleep(1) # Small delay before fallback
                    continue
                else:
                    # Non-retryable error (invalid prompt, etc.)
                    print(f"[ERROR] Non-retryable error with {model_name}: {e}")
                    raise e
        
        raise last_error or Exception("No models available to fulfill the request.")

    async def _call_image_gen(self, prompt):
        """Attempts to generate an image using available image models, falling back on failure."""
        last_error = None
        for model_name in self.image_models:
            try:
                print(f"[FALLBACK] Trying image model: {model_name}")
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=model_name,
                        contents=prompt
                    ),
                    timeout=60
                )
                return response, model_name
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["429", "resource_exhausted", "quota", "exhausted", "500", "internal"]):
                    print(f"[FALLBACK] Image model {model_name} failed: {e}. Trying next...")
                    last_error = e
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"[ERROR] Non-retryable error with image model {model_name}: {e}")
                    raise e
        raise last_error or Exception("No image models available.")

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
            # Record in memory ledger
            self.memory.add_experience(hist[-1][0], success, feedback)

    def update_last_history_result(self, chat_id: str, refined_result: str):
        chat_id_str = str(chat_id)
        hist = self.get_history(chat_id_str)
        if hist:
            hist[-1][2] = refined_result
            self.save_sessions()

    def reset_task_steps(self):
        """Resets the per-task step journal and screenshot buffer. Call this at the start of every new browser task."""
        self._task_steps = []
        self._task_screenshots = []
        self._last_action_fingerprint = ""
        self._stuck_count = 0
        self.current_plan = None # Clear plan for new task
        print("[STEP JOURNAL] Reset for new task.")

    def load_prompt(self):
        try:
            with open("system_prompt.txt", "r") as f:
                self.system_instruction = f.read()
        except FileNotFoundError:
            print("Warning: system_prompt.txt not found. Please ensure it exists.")
            self.system_instruction = ""
            
    async def decide_strategy(self, user_instruction: str, chat_id: int) -> tuple[str, str, str]:
        """
        Asks Gemini if the user instruction requires a browser or if it can be answered directly.
        Returns a tuple: (strategy, response_text, image_prompt). Strategy can be 'BROWSER', 'DIRECT', or 'IMAGE'.
        """
        chat_id_str = str(chat_id)
        chat_history = self.get_history(chat_id_str)
        history_context = ""
        if chat_history:
            history_context = "PREVIOUS INTERACTIONS:\n"
            for entry in chat_history:
                history_context += f"- User: {entry[0]}\n  Result: {entry[2]}\n"

        memory_context = self.memory.get_context_summary()

        decision_prompt = f"""
{history_context}
{memory_context}
LEARNED OPTIMIZATIONS:
{self.learned_optimizations}

USER REQUEST: {user_instruction}

Analyze the user request. Decide if you need to:
1. Use a web BROWSER to provide an accurate, up-to-date answer.
2. Answer DIRECTly using your internal knowledge (for general knowledge, math, conversation).
3. Generate an IMAGE.

Return a JSON object:
{{
  "thought": "Analyze the request and past experience here.",
  "strategy": "BROWSER" | "DIRECT" | "IMAGE",
  "reasoning": "short explanation",
  "direct_answer": "Your answer if strategy is DIRECT, otherwise null",
  "image_prompt": "Specific prompt for the image if strategy is IMAGE, otherwise null"
}}
"""
        try:
            response = await self._call_gemini(
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
            
            # If BROWSER, create a plan
            if strategy == "BROWSER":
                 self.current_plan = await self.planner.create_plan(user_instruction, history_context, self.learned_optimizations)
                 print(f"[PLANNER] Created plan: {json.dumps(self.current_plan, indent=2)}")

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
            # Prepare image parts if provided
            parts = [types.Part.from_text(text=verification_prompt)]
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))
            if user_image_path and os.path.exists(user_image_path):
                with open(user_image_path, "rb") as f:
                    parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))

            response = await self._call_gemini(
                contents=parts,
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
            response = await self._call_gemini(
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
                response = await self._call_gemini(
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

    async def analyze_and_act(self, user_instruction: str, screenshot_bytes: bytes, chat_id: int, user_image_path: str = None) -> tuple[str, bool, str]:
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
            history_context = "RECENT INTERACTIONS (last 5, use this context to inform your next actions):\n"
            for entry in chat_history[-5:]: # Only take the last 5 entries
                past_instruction = entry[0]
                past_action = entry[1] if len(entry) > 1 else ""
                past_result = entry[2] if len(entry) > 2 else ""
                history_context += f"- User: {past_instruction}\n  Action Taken: {past_action}\n  Result: {past_result}\n"
            history_context += "\n"

        memory_context = self.memory.get_context_summary()
        plan_context = f"\nCURRENT GLOBAL PLAN:\n{json.dumps(self.current_plan, indent=2)}\n" if self.current_plan else ""
        dom_snapshot = ""
        try:
            dom_snapshot = await self.browser.get_accessibility_snapshot()
        except:
            pass
        dom_context = f"\nDOM SNAPSHOT (Interactive Elements):\n{dom_snapshot}\n" if dom_snapshot else ""

        prompt_parts = [
            history_context, 
            memory_context,
            plan_context,
            dom_context,
            f"CURRENT USER INSTRUCTION: {user_instruction}",
            f"CURRENT STEP COUNT: {len(self._task_steps)}",
            "\nTASK REASONING GUIDELINE: Before acting, look at the screenshot, the DOM snapshot, the plan, and past failures. "
            "Think about whether your next action will bring you closer to the success criteria."
        ]
        
        # --- Per-task Step Journal (self-aware loop detection) ---
        # Inject every step taken so far in this browser task so Gemini can
        # see its own history and recognize if it is repeating itself.
        if self._task_steps:
            journal_lines = ["\nCURRENT TASK PROGRESS (your step-by-step actions so far for THIS task):"]
            for step in self._task_steps:
                actions_summary = ", ".join(
                    f"{a.get('action')}({a.get('text','')[:30] or a.get('coordinates','')})"
                    for a in step["actions"]
                )
                journal_lines.append(f"  Turn {step['turn']}: [URL: {step.get('url','?')}] {actions_summary}")
                if step.get("page_text"):
                    journal_lines.append(f"    → Page snippet: {step['page_text'][:150]}")
            
            # --- URL-based stuck detection ---
            # If the URL hasn't changed for 3+ consecutive turns, the bot is stuck.
            recent_urls = [s.get("url", "") for s in self._task_steps[-6:]]
            # Count trailing consecutive identical URLs
            url_stuck_count = 1
            for i in range(len(recent_urls) - 2, -1, -1):
                if recent_urls[i] == recent_urls[-1]:
                    url_stuck_count += 1
                else:
                    break
            
            print(f"[STUCK DETECTION] URL={recent_urls[-1]!r}, same for {url_stuck_count} consecutive turns.")
            
            if url_stuck_count >= 3:
                journal_lines.append(
                    f"\n  ⛔ CRITICAL STUCK ALERT: The page URL has been '{recent_urls[-1]}' for "
                    f"{url_stuck_count} consecutive turns. YOUR ACTIONS ARE HAVING NO EFFECT ON THE PAGE. "
                    "You MUST abandon the current approach and try something completely different:\n"
                    "  1. Try using browser's built-in form fill: use 'navigate' to the page URL again (hard refresh).\n"
                    "  2. After reload: click directly on input fields using fresh coordinates (try center of viewport).\n"
                    "  3. Submit using Tab key to move between fields, then Enter to submit.\n"
                    "  4. If still stuck, use 'read' to extract the page text and check for error messages.\n"
                    "  NEVER repeat the same click+type+click+type+key sequence the same way again."
                )
                if self.current_plan:
                   try:
                       self.current_plan = await self.planner.update_plan(self.current_plan, len(self._task_steps), f"Stuck on URL {recent_urls[-1]} for {url_stuck_count} turns.")
                       print(f"[RE-PLANNING] Updated plan: {json.dumps(self.current_plan, indent=2)}")
                   except Exception as plan_err:
                       print(f"[RE-PLANNING ERROR] {plan_err}")
            
            prompt_parts.append("\n".join(journal_lines))
        
        prompt = "\n".join(prompt_parts)
        
        # --- Manage Screenshot History ---
        # Add current screenshot to history (keep last 10)
        self._task_screenshots.append(screenshot_bytes)
        if len(self._task_screenshots) > 10:
            self._task_screenshots.pop(0)

        try:
            # Build content parts: text prompt + historical screenshots + current screenshot + optional user image
            parts = [types.Part.from_text(text=prompt)]
            
            # Add up to 3 previous screenshots for visual loop comparison
            # We don't want to overload with all 10, but 3 previous + 1 current is enough to see a stall.
            hist_frames = self._task_screenshots[-4:-1] # Get 3 frames before the current one
            for i, frame in enumerate(hist_frames):
                parts.append(types.Part.from_text(text=f"\nPREVIOUS SCREENSHOT {len(hist_frames)-i} STEPS AGO:"))
                parts.append(types.Part.from_bytes(data=frame, mime_type="image/jpeg"))
            
            parts.append(types.Part.from_text(text="\nCURRENT SCREENSHOT (THIS TURN):"))
            parts.append(types.Part.from_bytes(data=screenshot_bytes, mime_type="image/jpeg"))

            if user_image_path and os.path.exists(user_image_path):
                parts.append(types.Part.from_text(text="\nUSER PROVIDED REFERENCE IMAGE:"))
                with open(user_image_path, "rb") as f:
                    parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))

            response = await self._call_gemini(
                contents=[
                    types.Content(
                        role="user",
                        parts=parts,
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
                data = json.loads(response_text)
                # Ensure we are working with a list of actions
                if isinstance(data, dict):
                    if "actions" in data and isinstance(data["actions"], list):
                        actions_data = data["actions"]
                        thought = data.get("thought", "")
                        if thought:
                            print(f"\n[THOUGHT]: {thought}\n")
                    else:
                        actions_data = [data] # fallback
                elif isinstance(data, list):
                    actions_data = data
                else:
                    return "Error: Gemini did not return a valid action array or object.", False, None
            except json.JSONDecodeError:
                return f"Error: Gemini returned invalid JSON: {response_text}", False, None

            is_done = False
            final_result = "Processing..."
            
            # Execute multiple actions
            for action_data in actions_data:
                action = action_data.get("action")
                text = action_data.get("text", "")
                coordinates = action_data.get("coordinates")
                key = action_data.get("key", "")
                reasoning = action_data.get("reasoning", "")
                
                print(f"Executing Action: {action} ({reasoning})")
                
                if action == "navigate":
                    await self.browser.navigate(text)
                    await asyncio.sleep(1.5)  # Let page load after navigation
                elif action == "click":
                    if coordinates and len(coordinates) == 2:
                        await self.browser.click(int(coordinates[0]), int(coordinates[1]))
                        # Longer wait after submit-like clicks to allow page transitions
                        if any(kw in reasoning.lower() for kw in ["login", "iniciar", "submit", "sign in", "guardar", "save", "register", "registrar"]):
                            await asyncio.sleep(2.5)
                    else:
                        print(f"Warning: click action missing valid coordinates: {action_data}")
                elif action == "type":
                    # Use fill_field: triple-click to select all existing content, then type
                    # This REPLACES whatever is in the field instead of appending.
                    if coordinates and len(coordinates) == 2:
                        await self.browser.fill_field(int(coordinates[0]), int(coordinates[1]), text)
                    else:
                        await self.browser.type_text(text)
                # --- Semantic (coordinate-free) actions — PREFERRED for forms ---
                elif action == "fill_by_placeholder":
                    placeholder = action_data.get("placeholder", text)
                    result = await self.browser.fill_by_placeholder(placeholder, text)
                    print(f"  → {result}")
                elif action == "fill_by_label":
                    label = action_data.get("label", text)
                    result = await self.browser.fill_by_label(label, text)
                    print(f"  → {result}")
                elif action == "click_button":
                    result = await self.browser.click_by_text(text)
                    print(f"  → {result}")
                    if any(kw in text.lower() for kw in ["iniciar", "login", "sign in", "submit", "guardar", "registrar"]):
                        await asyncio.sleep(2.5)
                # --- SoM Actions ---
                elif action == "click_id":
                    som_id = action_data.get("id", text)
                    result = await self.browser.click_by_id(som_id)
                    print(f"  → {result}")
                elif action == "fill_id":
                    som_id = action_data.get("id")
                    result = await self.browser.fill_by_id(som_id, text)
                    print(f"  → {result}")
                elif action == "inspect_form":
                    fields_json = await self.browser.get_form_fields()
                    final_result = f"Form fields: {fields_json}"
                    print(f"  → {final_result[:200]}")
                elif action == "key":
                    await self.browser.press_key(key or text)
                elif action == "read":
                    page_text = await self.browser.get_text_content()
                    final_result = page_text[:2000] if page_text else "No text found."
                elif action == "scroll":
                    direction = action_data.get("direction", "down").lower()
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
                
                # Small delay between chained actions to let the page update
                await asyncio.sleep(0.5)
            
            # --- Record this turn in the step journal ---
            page_text_snippet = ""
            current_url = ""
            try:
                page_text_snippet = await self.browser.get_text_content()
                page_text_snippet = page_text_snippet[:300] if page_text_snippet else ""
            except Exception:
                pass
            try:
                current_url = await self.browser.get_url()
            except Exception:
                pass
            
            self._task_steps.append({
                "turn": len(self._task_steps) + 1,
                "actions": [
                    {"action": d.get("action"), "text": d.get("text", ""), 
                     "coordinates": d.get("coordinates"), "reasoning": d.get("reasoning", "")}
                    for d in actions_data
                ],
                "page_text": page_text_snippet,
                "url": current_url,
            })
            print(f"[STEP JOURNAL] Turn {len(self._task_steps)} recorded. URL={current_url!r}")
            
            # --- Hard bailout: if the same non-empty URL has appeared 4+ times total ---
            if not is_done and current_url:
                url_total_count = sum(1 for s in self._task_steps if s.get("url") == current_url)
                if url_total_count >= 4:
                    bail_msg = (
                        f"🛑 I've been stuck on '{current_url}' for {url_total_count} turns "
                        f"(out of {len(self._task_steps)} total). "
                        "My actions are having no lasting effect on the page. "
                        "Possible causes: wrong credentials, a CAPTCHA, JavaScript blocking input, or a page error. "
                        "Please check the page and try again."
                    )
                    print(f"[STUCK BAILOUT] URL '{current_url}' seen {url_total_count} times. Aborting.")
                    return bail_msg, True, None

            
            # Save to history only when the task is done
            if is_done:
                self._task_steps = []  # Reset journal on task completion
                self._last_action_fingerprint = ""  # Reset loop state on completion
                self._stuck_count = 0
                self.add_to_history(chat_id_str, user_instruction, actions_data, final_result)
            
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

            return final_result, is_done, tts_path

        except Exception as e:
            print(f"Error in analyze_and_act: {e}")
            return f"Error: {e}", True, None

    async def generate_image(self, prompt: str) -> str:
        """
        Uses available image models to generate an image from a prompt.
        """
        try:
            # Add a 60-second timeout to prevent indefinite hangs
            response, model_used = await self._call_image_gen(prompt)
            
            # Extract binary image data
            image_bytes = None
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if "image" in part.inline_data.mime_type:
                                image_bytes = part.inline_data.data
                                break
            
            if image_bytes:
                filename = "generated_image.jpg"
                with open(filename, "wb") as f:
                    f.write(image_bytes)
                return f"IMAGE:{filename}"
            
            # Fallback to response text if no image data found
            if response.text:
                return f"Generated Content: {response.text}"
                
            return "Error: Model returned no image data or text."
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return f"Error generating image: {str(e)}"

    async def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribes the audio file using Gemini.
        Returns the transcribed text.
        """
        try:
            # Upload the file using the modern genai SDK
            audio_file = await self.client.aio.files.upload(file=audio_file_path)
            
            prompt = "Transcribe this audio exactly as it is spoken. Do not include anything else in your response."
            
            response = await self._call_gemini(
                contents=[audio_file, prompt],
                config=types.GenerateContentConfig()
            )
            
            # Cleanup uploaded file from Gemini storage
            await self.client.aio.files.delete(name=audio_file.name)
            
            return response.text.strip()
            
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"
