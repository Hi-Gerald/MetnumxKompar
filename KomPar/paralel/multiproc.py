from multiprocessing import Process
import time

def test_process(process_id, sleep_time, a, b):
    print(f"Thread {process_id} mulai pada {time.perf_counter():.4f}")
    a = a + 1
    b = b + 1
    a = a * b
    b = a / b
    time.sleep(sleep_time)
    print(f"Thread {process_id} selesai pada {time.perf_counter():.4f}")

if __name__ == '__main__':
    num_processes = 10
    a = 2
    b = 3
    processes = []

    start = time.perf_counter()

    for i in range(num_processes):
        process = Process(target=test_process, args=(i, 5, a, b))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    end = time.perf_counter()
    print(f"Total waktu eksekusi: {end - start:.4f} detik")
    print("Semua proses selesai.")
