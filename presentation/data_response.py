from typing import List


class DataResponse:
    def __init__(self, id: str, token: str | List[float], is_done: bool):
        self.id = id
        self.token = token
        self.is_done = is_done
