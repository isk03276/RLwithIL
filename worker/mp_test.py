from multiprocessing import Manager, managers, Queue, Array
from multiprocessing import Process


class Buffer:
    def __init__(self):
        self.buffer = []

    def append(self, i):
        self.buffer.append(i)

    def test(self, shared_queue):
        data = [i for i in range(10)]
        shared_queue.put(data)


if __name__ == "__main__":
    shared_queue = Queue()
    buffer = Buffer()

    p = Process(target=buffer.test, args=(shared_queue,))
    p.start()
    p.join()
    print(shared_queue.get())
