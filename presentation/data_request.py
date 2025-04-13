from typing import List
from core.conversation import Conversation
from core.supported_tasks import SupportedTasks
from presentation.data_response import DataResponse


class DataRequest:
    def __init__(self, id: str, conversation: Conversation, task: SupportedTasks):
        self.id = id
        self.conversation = conversation
        self.task = task
        self.output_tokens: List[str] = ['']
        self.is_done = False

    def add_response(self, data_response: DataResponse) -> None:
        if data_response.id != self.id:
            raise Exception('Unknown error')

        self.output_tokens.append(data_response.token)
        self.is_done = data_response.is_done
