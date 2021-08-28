from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, algo_name, env_name, file_directory="log/"):
        self.file_path = file_directory + algo_name + env_name
        self.writer = SummaryWriter(self.file_path)
        print(self.file_path)

    def log(self, name, data, step):
        self.writer.add_scalar(name, data, step)
