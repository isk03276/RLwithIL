class SingleWorker:
    def rollout(self):
        i = 0
        while True:
            i += 1
            if i % 4 == 0:
                yield i


if __name__ == "__main__":
    single_worker = SingleWorker()
    gen_func = single_worker.rollout()
    print(gen_func)
    input("go?")
    while True:
        i = next(gen_func)
        print(i)
