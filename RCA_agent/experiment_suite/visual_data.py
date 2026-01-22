import json

class VisualDataLogger:
    def __init__(self, log_file="viz_data.jsonl"):
        self.log_file = log_file

    def log_thinking_process(self, turn_id, model, state, strategy, action):
        entry = {
            "type": "sankey_flow",
            "turn_id": turn_id,
            "model": model,
            "flow": {
                "source": state,
                "target": strategy,
                "next": "Response"
            }
        }
        self._write(entry)

    def log_reward_surface(self, turn_id, candidates):
        scores = [c.get('iva_score', 0) for c in candidates]
        entry = {
            "type": "reward_dist",
            "turn_id": turn_id,
            "scores": scores
        }
        self._write(entry)

    def _write(self, data):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")