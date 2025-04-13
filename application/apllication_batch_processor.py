from typing import Dict, List
from application.model_input_sample import ModelInputSample
from infrastructure.auto_regressive_text_model import AutoRegressiveTextModel
from presentation.data_request import DataRequest
from presentation.data_response import DataResponse
from infrastructure.auto_regressive_text_batch import AutoRegressiveTextBatch
from core.supported_tasks import SupportedTasks
import torch


class ApplicationBatchProcessor:
    def __init__(self, model: AutoRegressiveTextModel, max_batch_size: int, ):
        model.load_model()
        self.max_batch_size = max_batch_size
        self.model = model
        self.samples: Dict[str, ModelInputSample] = {}
        self.samples_ids_queue: List[str] = []

    def add_sample(self, data_request: DataRequest) -> None:
        self.samples[data_request.id] = ModelInputSample(
            id=data_request.id, conversation=data_request.conversation, task=data_request.task)
        self.samples_ids_queue.append(data_request.id)

    def get_batch(self,) -> AutoRegressiveTextBatch:
        input_batch: AutoRegressiveTextBatch = AutoRegressiveTextBatch.empty()

        for index in range(min(len(self.samples_ids_queue), self.max_batch_size)):
            input_batch.ids.append(
                self.samples[self.samples_ids_queue[index]].id)
            input_batch.conversations_list.append(
                self.samples[self.samples_ids_queue[index]].conversation)
            input_batch.input_ids_list.append(
                self.samples[self.samples_ids_queue[index]].input_ids)
            input_batch.attention_mask_list.append(
                self.samples[self.samples_ids_queue[index]].attention_mask)
            input_batch.past_key_values.append(None)
            input_batch.embedding_list.append(
                self.samples[self.samples_ids_queue[index]].embeddings)
            input_batch.is_done_list.append(
                self.samples[self.samples_ids_queue[index]].is_done)
            input_batch.next_tokens_ids_list.append(None)
            input_batch.next_tokens_list.append(None)
            self.samples[self.samples_ids_queue[index]].clear_old_data()

        return input_batch

    def generate(self,) -> List[DataResponse]:

        if len(self.samples) == 0:
            return []

        input_batch = self.get_batch()

        input_batch = self.model(auto_regressive_batch=input_batch)

        responses_list: List[DataResponse] = []

        for index, sample_id in enumerate(input_batch.ids):
            is_done = input_batch.is_done_list[index] or self.samples[sample_id].task == SupportedTasks.embedding

            self.samples[sample_id].input_ids = torch.concat(
                (input_batch.input_ids_list[index], input_batch.next_tokens_ids_list[index].unsqueeze(0)), dim=-1)

            self.samples[sample_id].attention_mask = torch.concat(
                (input_batch.attention_mask_list[index], torch.tensor([1], device=input_batch.attention_mask_list[index].device, dtype=input_batch.attention_mask_list[index].dtype)), dim=-1)

            self.samples[sample_id].past_key_values = input_batch.past_key_values[index]

            self.samples[sample_id].embeddings = input_batch.embedding_list[index]
            self.samples[sample_id].is_done = is_done

            if self.samples[sample_id].task == SupportedTasks.embedding:
                responses_list.append(DataResponse(
                    id=sample_id, token=input_batch.embedding_list[index], is_done=True))
            else:
                responses_list.append(DataResponse(
                    id=sample_id, token='' if is_done else input_batch.next_tokens_list[index], is_done=is_done))

            if is_done:
                del self.samples[sample_id]
                self.samples_ids_queue.remove(sample_id)

        input_batch = None

        return responses_list
