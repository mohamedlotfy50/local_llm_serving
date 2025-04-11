from core.message import Message
from typing import List, Dict


class Conversation:
    def __init__(self, messages: List[Message]):
        self.messages = messages

    def from_json(json_list: List) -> 'Conversation':
        messages = []

        for json_data in json_list:
            messages.append(Message.from_json(json_data=json_data))

        return Conversation(messages=messages)

    def to_json(self,) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        for message in self.messages:
            messages.append(message.to_json())

        return messages
