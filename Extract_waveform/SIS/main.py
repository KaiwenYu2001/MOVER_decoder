import shutil
from multiprocessing import Pool
import Example_Of_Execution
import os
import time


USE_MULTITHREADING = True
MAX_POOL_SIZE = 10
INPUT_DIR= "/Users/kevinyu/Desktop/test"
OUTPUT_DIR = "/Users/kevinyu/Desktop/result"


class AccumulatedLogger:
    accumulated_logs: list = []

    def info(self, msg: str, prefix="[INFO] ") -> None:
        self.accumulated_logs.append(f"{prefix}{msg}")

    def warn(self, msg: str, prefix="[WARN] ") -> None:
        self.accumulated_logs.append(f"\x1b[33m{prefix}{msg}\x1b[0m")

    def error(self, msg: str, prefix="[ERROR] ") -> None:
        self.accumulated_logs.append(f"\x1b[31m{prefix}{msg}\x1b[0m")

    def flush(self) -> str:
        tmp = '\n'.join(self.accumulated_logs)
        tmp = tmp + "\n"
        return tmp

    def flush_and_print_all(self) -> None:
        print(self.flush())


class Timer:
    total_time: float = 0.0
    current_time = None

    def start(self):
        self.current_time = time.time()

    def end(self) -> float:
        elasped = time.time() - self.current_time
        self.total_time += elasped
        self.current_time = None
        return elasped


def process(patient_dir):
    timer = Timer()
    timer.start()
    log = AccumulatedLogger()
    patient_id = os.path.split(patient_dir)[1]
    try:
        log.info(f"Patient ID: {patient_id}")
        Example_Of_Execution.Execute(patient_dir, OUTPUT_DIR)
        log.info(f"success, finished in : {timer.end()}s")
    except Exception as err:
        log.error(f"Error during wave assembly (patient_id={patient_id}): {err}")
        log.info(f"Finished in : {timer.end()}s")
    log.flush_and_print_all()


def main():
    timer = Timer()
    timer.start()

    if not os.path.exists(OUTPUT_DIR):
        print("Make New Dir...\n")
        os.mkdir(OUTPUT_DIR)
    else:
        print("Remove file in Dir...\n")
        shutil.rmtree(OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    patient_dirs = []
    for PID in os.listdir(INPUT_DIR):
        if PID[0] != '.':
            patient_dirs.append(os.path.join(INPUT_DIR, PID))

    if USE_MULTITHREADING:
        with Pool(MAX_POOL_SIZE) as pool:
            pool.map(process, patient_dirs)
    else:
        for patient_dir in patient_dirs:
            process(patient_dir)
    print(
        f"Total time elapsed for all {len(patient_dirs)} patients: {timer.end()}s")


if __name__ == '__main__':
    main()
