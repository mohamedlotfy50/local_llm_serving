from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from infrastructure.auto_regressive_text_batch import AutoRegressiveTextBatch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import List, Dict, Tuple
from infrastructure.padding_side import PaddingSide
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


class AutoRegressiveTextModel:
    def __init__(self, model_path: str, padding_side: PaddingSide = None, device: str = 'cuda'):
        self.device = "cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu"
        self.model_path = model_path

        self.required_padding_side = padding_side

        self.model: Qwen2ForCausalLM = None

        self.tokenizer: Qwen2TokenizerFast = None

    def load_model(self, ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device)

        self.padding_side: PaddingSide = PaddingSide(
            self.tokenizer.padding_side) if self.required_padding_side is None else self.required_padding_side

        self.bos_token_id: int = self.model.generation_config.bos_token_id

        self.pad_token_id: int = self.model.generation_config.pad_token_id

        self.do_sample: int = self.model.generation_config.do_sample

        self.eos_token_ids: List[int] = self.model.generation_config.eos_token_id

        self.repetition_penalty: int = self.model.generation_config.repetition_penalty

        self.temperature: int = self.model.generation_config.temperature

        self.top_p: int = self.model.generation_config.top_p

        self.top_k: int = self.model.generation_config.top_k

    def __call__(self,  auto_regressive_batch: AutoRegressiveTextBatch) -> AutoRegressiveTextBatch:
        auto_regressive_batch = self.process_batch_input(
            auto_regressive_batch=auto_regressive_batch)

        auto_regressive_batch = self.remove_batch_padding(
            auto_regressive_batch=auto_regressive_batch)

        auto_regressive_batch = self.apply_batch_padding(
            auto_regressive_batch=auto_regressive_batch)

        auto_regressive_batch = self.forward(
            auto_regressive_batch=auto_regressive_batch)
        auto_regressive_batch = self.remove_batch_padding(
            auto_regressive_batch=auto_regressive_batch)

        return auto_regressive_batch

    def process_batch_input(self, auto_regressive_batch: AutoRegressiveTextBatch) -> AutoRegressiveTextBatch:

        conversations: List[List[Dict[str, str]]] = []
        conversation_indecis: List[List[Dict[str, str]]] = []

        for index, conversation in enumerate(auto_regressive_batch.conversations_list):
            if auto_regressive_batch.input_ids_list[index] is None:
                conversations.append(conversation.to_json())
                conversation_indecis.append(index)

        if len(conversations) == 0:

            return auto_regressive_batch

        chat_list = self.tokenizer.apply_chat_template(conversations,
                                                       tools=None,
                                                       truncation=False,
                                                       tokenize=False,
                                                       add_generation_prompt=True,)

        model_inputs = self.tokenizer(
            chat_list, truncation=False, padding=True, padding_side=self.padding_side.value, return_tensors="pt")

        for index, position in enumerate(conversation_indecis):
            auto_regressive_batch.input_ids_list[position] = model_inputs.input_ids[index]
            auto_regressive_batch.attention_mask_list[position] = model_inputs.attention_mask[index]

        return auto_regressive_batch

    def _apply_tensor_padding(self, torch_input: torch.Tensor, padding_size: int, padding_value: int) -> torch.Tensor:
        input_padding = torch.full(
            (padding_size,), padding_value, device=torch_input.device, dtype=torch_input.dtype)

        if self.padding_side == PaddingSide.right:
            return torch.concat((torch_input, input_padding), dim=-1)

        return torch.concat((input_padding, torch_input), dim=-1)

    def _remove_tensor_padding(self, torch_input: torch.Tensor, padding_token: int) -> torch.Tensor:
        padding_size = (torch_input == padding_token).sum()

        if self.padding_side == PaddingSide.right:
            torch_input_size = torch_input.size(-1)
            return torch_input[:torch_input_size-padding_size]

        return torch_input[padding_size:]

    def remove_batch_padding(self, auto_regressive_batch: AutoRegressiveTextBatch) -> AutoRegressiveTextBatch:
        for index, input_ids in enumerate(auto_regressive_batch.input_ids_list):
            input_ids = self._remove_tensor_padding(
                torch_input=input_ids, padding_token=self.pad_token_id)

            attention_mask = self._remove_tensor_padding(
                torch_input=auto_regressive_batch.attention_mask_list[index], padding_token=0)

            auto_regressive_batch.input_ids_list[index] = input_ids
            auto_regressive_batch.attention_mask_list[index] = attention_mask

        return auto_regressive_batch

    def apply_batch_padding(self, auto_regressive_batch: AutoRegressiveTextBatch) -> AutoRegressiveTextBatch:

        max_size = max([attention_mask.sum()
                       for attention_mask in auto_regressive_batch.attention_mask_list])

        for index, input_ids in enumerate(auto_regressive_batch.input_ids_list):
            if input_ids.size(-1) < max_size:

                padded_input_ids = self._apply_tensor_padding(
                    torch_input=input_ids,
                    padding_size=max_size - input_ids.size(-1),
                    padding_value=self.pad_token_id)

                padded_attention_mask = self._apply_tensor_padding(
                    torch_input=auto_regressive_batch.attention_mask_list[index],
                    padding_size=max_size -
                    auto_regressive_batch.attention_mask_list[index].size(-1),
                    padding_value=0)

                auto_regressive_batch.input_ids_list[index] = padded_input_ids
                auto_regressive_batch.attention_mask_list[index] = padded_attention_mask

        return auto_regressive_batch

    def forward(self, auto_regressive_batch: AutoRegressiveTextBatch) -> AutoRegressiveTextBatch:

        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=torch.stack(auto_regressive_batch.input_ids_list).to(
                    self.model.device),
                attention_mask=torch.stack(
                    auto_regressive_batch.attention_mask_list).to(self.model.device),
                past_key_values=None,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )

        next_tokens_ids_batch = torch.argmax(
            torch.softmax(outputs.logits.cpu()[:, -1, :], dim=-1), dim=-1)

        next_tokens = self.tokenizer.batch_decode(next_tokens_ids_batch)

        for index in range(len(auto_regressive_batch.ids)):

            auto_regressive_batch.embedding_list[index] = torch.mean(outputs.hidden_states[-1][index]
                                                                     [auto_regressive_batch.attention_mask_list[index].bool(), :], dim=0).cpu().tolist()
            auto_regressive_batch.next_tokens_ids_list[index] = next_tokens_ids_batch[index]
            auto_regressive_batch.is_done_list[index] = next_tokens_ids_batch[index] in self.eos_token_ids
            auto_regressive_batch.next_tokens_list[index] = next_tokens[index]

        return auto_regressive_batch
