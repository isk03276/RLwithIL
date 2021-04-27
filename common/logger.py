from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, file_name, file_directory="log/"):
        self.file_path = file_directory + file_name
        self.writer = SummaryWriter(self.file_path)

    def log(self, name, data, step):
        self.writer.add_scalar(name, data, step)

