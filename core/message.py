from typing import Dict


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def from_json(json_data) -> 'Message':

        return Message(role=json_data['role'], content=json_data['content'])

    def to_json(self) -> Dict[str, str]:
        return {
            'role': self.role,
            'content': self.content
        }
