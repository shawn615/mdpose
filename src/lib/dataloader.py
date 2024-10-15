import math
import time
import torch
import multiprocessing
import threading


class DataLoader(multiprocessing.Process):
    # multi thread data loader
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=1):
        multiprocessing.Process.__init__(self)

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.max_q_size = num_workers * 5
        self.batch_q = multiprocessing.Queue(self.max_q_size)
        self.n_samples = self.dataset.__len__()

        self.daemon = True
        self.data_idx = 0
        self.token = 0

    def __len__(self):
        return math.ceil(self.n_samples / self.batch_size)

    def __load_batch__(self, thread_num, data_idx):
        batch_dict = dict()
        for i in range(self.batch_size):
            sample_dict = self.dataset.__getitem__((data_idx + i) % self.n_samples)
            for key, value in sample_dict.items():
                if key not in batch_dict.keys():
                    batch_dict[key] = list()
                batch_dict[key].append(torch.from_numpy(value))

        for key, value in batch_dict.items():
            batch_dict[key] = torch.stack(value, dim=0)

        while thread_num != self.token:
            time.sleep(0.001)
        self.batch_q.put(batch_dict)
        self.token = (self.token + 1) % self.num_workers
        # print("Thread num: %d, Data index: %d" % (thread_num, data_idx))

    def run(self):
        if self.shuffle:
            self.dataset.shuffle()
        thread_nums = range(self.num_workers)

        while True:
            threads = list()

            if self.batch_q.qsize() < (self.max_q_size - self.num_workers):
                for thread_num in thread_nums:
                    thread = threading.Thread(target=self.__load_batch__, args=(thread_num, self.data_idx))
                    thread.daemon = True
                    threads.append(thread)
                    self.data_idx = (self.data_idx + self.batch_size) % self.n_samples

                for thread_num in thread_nums:
                    threads[thread_num].start()
                for thread_num in thread_nums:
                    threads[thread_num].join()
            else:
                time.sleep(0.001)

    def get_batch(self):
        # print('get batch, q-size:', self.batch_q.qsize())
        return self.batch_q.get()

    def stop(self):
        self.terminate()
        self.join()

        while not self.batch_q.empty():
            # print('get batch, q-size:', self.batch_q.qsize())
            self.batch_q.get()
        print('get batch, q-size:', self.batch_q.qsize())
        self.batch_q.close()
        self.batch_q.join_thread()

        # while not self.batch_q.empty():
        #     # print('get batch, q-size:', self.batch_q.qsize())
        #     self.batch_q.get()
        # print('empty_q', self.batch_q.qsize())

