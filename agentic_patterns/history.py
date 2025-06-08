class ChatHistory(list):
    def __init__(self, messages: list | None = None, total_length: int = -1, popout_index: int = 0):
        if messages is None:
            messages = []

        super().__init__(messages)
        self.total_length = total_length
        self.popout_index = popout_index

    def append(self, msg: str):
        if len(self) == self.total_length:
            self.pop(self.popout_index)
        super().append(msg)
