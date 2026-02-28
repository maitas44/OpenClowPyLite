import json
from google.genai import types

class Planner:
    def __init__(self, agent):
        self.agent = agent

    async def create_plan(self, user_instruction, history_summary, learned_optimizations):
        """
        Decomposes a user instruction into a sequence of high-level steps.
        """
        planning_prompt = f"""
USER REQUEST: {user_instruction}

HISTORY CONTEXT:
{history_summary}

LEARNED OPTIMIZATIONS:
{learned_optimizations}

Break down the user request into a logical plan of action. 
Think step-by-step. What information do we need? What pages should we visit? 
What constitutes "success" for this task?

Return a JSON object:
{{
  "thought": "Deep reasoning about the task and potential challenges.",
  "plan": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "estimated_steps": 5,
  "success_criteria": "Directly answering the question about X with data from Y."
}}
"""
        try:
            response = await self.agent._call_gemini(
                contents=planning_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[PLANNER] Error creating plan: {e}")
            return {
                "thought": "Fallback planning due to error.",
                "plan": ["Execute browser loop directly."],
                "estimated_steps": 10,
                "success_criteria": "Fulfill request as best as possible."
            }

    async def update_plan(self, current_plan, current_step_index, feedback):
        """
        Adjusts the plan based on feedback from a failed verification or a stuck state.
        """
        update_prompt = f"""
CURRENT PLAN: {json.dumps(current_plan)}
CURRENT STEP INDEX: {current_step_index}
FEEDBACK/ISSUE: {feedback}

The bot is having trouble or a verification failed. Adjust the plan to overcome this obstacle.

Return the updated plan as a JSON object:
{{
  "thought": "Analysis of why we got stuck and how the plan fixes it.",
  "updated_plan": [...],
  "new_step_index": 0
}}
"""
        try:
            response = await self.agent._call_gemini(
                contents=update_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[PLANNER] Error updating plan: {e}")
            return current_plan
