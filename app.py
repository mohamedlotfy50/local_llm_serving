from flask import Flask, request, jsonify, stream_with_context, Response
from typing import Dict
import uuid
from core.conversation import Conversation
import threading
from infrastructure.auto_regressive_text_model import AutoRegressiveTextModel
from huggingface_hub import snapshot_download
import torch
from presentation.data_request import DataRequest
from application.apllication_batch_processor import ApplicationBatchProcessor
from core.supported_tasks import SupportedTasks
from core.message import Message
import json

model_path = "assets\\Qwen2_0.5B_Instruct"
max_batch_size = 2

model = AutoRegressiveTextModel(model_path=model_path)

app_processor = ApplicationBatchProcessor(
    max_batch_size=max_batch_size, model=model, )

app = Flask(__name__)


data_dict: Dict[str, DataRequest] = {}


@app.route('/complete', methods=['POST'])
def complete():
    data = request.get_json()
    request_id = str(uuid.uuid4())
    conversation = Conversation.from_json(data)
    data_request = DataRequest(
        id=request_id, conversation=conversation, task=SupportedTasks.completion)
    data_dict[request_id] = data_request
    app_processor.add_sample(data_request)

    return Response(stream_with_context(stream_data(request_id)), mimetype='application/json')


def stream_data(request_id):
    while not data_dict[request_id].is_done:
        if len(data_dict[request_id].output_tokens) > 0:
            token = data_dict[request_id].output_tokens.pop(0)
            message = Message(role='assistant', content=token)
            yield json.dumps(message.to_json()) + '\n'

    del data_dict[request_id]


def llm_model():
    while True:
        respons_data_list = app_processor.generate()
        for respons_data in respons_data_list:
            data_dict[respons_data.id].add_response(respons_data)


if __name__ == '__main__':

    x = threading.Thread(target=llm_model)
    # x.start()
    # run flsk in diferent thered
    # x = threading.Thread(target=lambda: app.run())

    try:
        x.start()
        app.run()
    except:
        pass
