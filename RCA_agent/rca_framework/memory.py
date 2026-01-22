from collections import deque

class SemanticMemory:
    def __init__(self, anchor_summary="", max_recent=15):
        self.m_anchor = anchor_summary
        self.m_recent = deque(maxlen=max_recent)
        self.turn_counter = 0

    def update_recent(self, turn):
        self.m_recent.append(turn)
        self.turn_counter += 1

    def get_phase_hint(self):
        if self.turn_counter < 3: return "Opening"
        if self.turn_counter > 25: return "Closing"
        return "Main Activity"

    def get_agent_context(self):
        transcript = "\n".join([f"{t.get('speaker_id')}: {t.get('text')}" for t in self.m_recent])
        return f"""
[Session Phase]: {self.get_phase_hint()}
[长者画像]
{self.m_anchor}
[对话记录]
{transcript}
"""

    def get_assessment_context(self):
        transcript = "\n".join([f"{t.get('speaker_id')}: {t.get('text')}" for t in self.m_recent])
        return f"""
[长者画像]
{self.m_anchor}
[对话记录]
{transcript}
"""