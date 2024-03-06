"""
This script handles all patients in one directory, that is: for any class like '01' ~ 'ff', this script
converts all data in that directory into csv waveform that is stored in the output directory.
"""

# ================================================================================
USE_MULTITHREADING = True
MAX_POOL_SIZE = 12
INPUT_DIR= "/Users/kevinyu/Desktop/test_env/test_case/epic_test"
OUTPUT_DIR = "/Users/kevinyu/Desktop/test_env/output"
# ================================================================================


import shutil
from multiprocessing import Pool
# import multiprocessing
import Example_Of_Execution
import os
import time


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


def process(patient_dir, patient_id):
    timer = Timer()
    timer.start()
    log = AccumulatedLogger()
    # log = multiprocessing.get_logger()
    try:
        log.info(f"Processing patient ID: {patient_id}")
        Example_Of_Execution.Execute(patient_dir, patient_id,  OUTPUT_DIR, debug=False)
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

    for starter_PID in os.listdir(INPUT_DIR):
        '''
        This is where this script different from main.py, since it should handle all patients
        '''
        if starter_PID[0] == '.':
            continue
        full_dir = os.path.join(INPUT_DIR, starter_PID) # 00, 01 here.
        pid_list = []
        for xmlname in os.listdir(full_dir):
            if xmlname[0] != '.' and xmlname.split('-')[0] not in pid_list:
                pid_list.append(xmlname.split('-')[0])

        patient_dirs = [(full_dir, pid) for pid in pid_list]
        print(patient_dirs)
        # dir_list = []
        # for i in range(len(pid_list)):
        #     dir_list.append(full_dir)
        if USE_MULTITHREADING:
            with Pool(MAX_POOL_SIZE) as pool:
                pool.starmap(process, patient_dirs)
        else:
            for patient_dir in patient_dirs:
                process(patient_dir[0], patient_dir[1])

    print(f"Total time elapsed for all patients: {timer.end()}s")


if __name__ == '__main__':
    main()
