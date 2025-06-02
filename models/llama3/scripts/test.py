import multiprocessing
import time

def worker(n):
    print(f"Worker {n} started")
    time.sleep(10)
    print(f"Worker {n} finished")

if __name__ == "__main__":
    print("Sleep 10 sec before launch new processes")
    time.sleep(10)
    p1 = multiprocessing.Process(target=worker, args=(1,))
    p2 = multiprocessing.Process(target=worker, args=(2,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
