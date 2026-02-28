import json
import os

class Memory:
    def __init__(self, sessions_file="sessions.json"):
        self.sessions_file = sessions_file
        self.ledger_file = "experience_ledger.json"
        self.experience = self._load_ledger()

    def _load_ledger(self):
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, "r") as f:
                    return json.load(f)
            except:
                return {"successes": [], "failures": []}
        return {"successes": [], "failures": []}

    def _save_ledger(self):
        with open(self.ledger_file, "w") as f:
            json.dump(self.experience, f, indent=2)

    def add_experience(self, instruction, success, feedback):
        entry = {
            "instruction": instruction,
            "feedback": feedback
        }
        if success:
            self.experience["successes"].append(entry)
            self.experience["successes"] = self.experience["successes"][-20:] # Keep last 20
        else:
            self.experience["failures"].append(entry)
            self.experience["failures"] = self.experience["failures"][-20:]
        self._save_ledger()

    def get_context_summary(self):
        """Returns a condensed summary of past successes and failures for the prompt."""
        summary = "PAST EXPERIENCE SUMMARY:\n"
        if self.experience["failures"]:
            summary += "Common Pitfalls (Avoid these):\n"
            for f in self.experience["failures"][-5:]:
                summary += f"- Failed on: '{f['instruction']}'. Reason: {f['feedback']}\n"
        
        if self.experience["successes"]:
            summary += "Successful Strategies:\n"
            for s in self.experience["successes"][-3:]:
                summary += f"- Succeeded on: '{s['instruction']}'\n"
        
        return summary if len(summary) > 25 else "No significant experience recorded yet."
