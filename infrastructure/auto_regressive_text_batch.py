from core.conversation import Conversation
from core.continous_kv_chache import ContinousKVCache
from typing import List
import torch


class AutoRegressiveTextBatch:
    def __init__(self, ids: List[str],
                 conversations_list: List[Conversation],
                 input_ids_list: List[torch.Tensor],
                 attention_mask_list: List[torch.Tensor],
                 past_key_values: List[ContinousKVCache],
                 embedding_list: List[List[float]],
                 next_tokens_ids_list: List[torch.Tensor],
                 next_tokens_list: List[str],
                 is_done_list: List[bool],):
        self.ids = ids
        self.conversations_list = conversations_list
        self.input_ids_list = input_ids_list
        self.attention_mask_list = attention_mask_list
        self.past_key_values = past_key_values
        self.embedding_list = embedding_list
        self.next_tokens_ids_list = next_tokens_ids_list
        self.next_tokens_list = next_tokens_list
        self.is_done_list = is_done_list

    def empty() -> 'AutoRegressiveTextBatch':
        return AutoRegressiveTextBatch(ids=[],
                                       conversations_list=[],
                                       attention_mask_list=[],
                                       input_ids_list=[],
                                       past_key_values=[],
                                       embedding_list=[],
                                       is_done_list=[],
                                       next_tokens_ids_list=[],
                                       next_tokens_list=[],)
