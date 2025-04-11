from core.conversation import Conversation
from core.supported_tasks import SupportedTasks
import torch
from core.continous_kv_chache import ContinousKVCache
from typing import List


class ModelInputSample:
    def __init__(self, id: str, conversation: Conversation,
                 task: SupportedTasks,
                 input_ids: torch.Tensor = None,
                 attention_mask: torch.Tensor = None,
                 past_key_values: ContinousKVCache = None,
                 embeddings: List[float] = None,
                 is_done: bool = False

                 ):
        self.id = id
        self.conversation = conversation
        self.task = task
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.past_key_values = past_key_values
        self.embeddings = embeddings
        self.is_done = is_done

    def clear_old_data(self,) -> None:
        self.input_ids = None
        self.attention_mask = None
        self.embeddings = None
        self.past_key_values = None
