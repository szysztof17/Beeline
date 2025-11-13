import time
import os
import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import getpass

CURRENT_USER = getpass.getuser()
BEELINE_OUTPUT_ROOT = "/home/kl467102/thesis/BEELINE/outputs/pbmc10k_meta"
TARGET_SUBDIRS = ["GENIE3", "GRNBOOST2"]
TARGET_FILENAME = "outFile.txt"
PROCESS_KEYWORDS = ["GENIE3", "GRNBOOST2"]


class GRNFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        rel_path = os.path.relpath(event.src_path, BEELINE_OUTPUT_ROOT)

        for method in TARGET_SUBDIRS:
            expected_path = os.path.join(method, TARGET_FILENAME)
            if rel_path.endswith(expected_path):
                dataset_path = os.path.dirname(os.path.dirname(event.src_path))
                dataset_rel_path = os.path.relpath(dataset_path, BEELINE_OUTPUT_ROOT)
                print(f"[Watcher] Detected {method} outFile at {event.src_path}")
                time.sleep(10)
                kill_grn_process_for_dataset(dataset_rel_path, method)
                break


def kill_grn_process_for_dataset(dataset_path, method):
    print(f"[Watcher] Attempting to kill {method} processes for dataset: {dataset_path}")
    killed_any = False

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd', 'username']):
        try:
            if proc.info['username'] == CURRENT_USER:
                cmdline_str = ' '.join(proc.info.get('cmdline', [])) or ''
                cwd = proc.info.get('cwd') or ''
                if method.lower() in cmdline_str.lower():
                    if dataset_path in cmdline_str or dataset_path in cwd:
                        if not "time" in cmdline_str.lower():
                            print(f"[Watcher] Killing {method} process {proc.pid} for dataset {dataset_path}")
                            print(f"    └─ CMD: {cmdline_str}")
                            print(f"    └─ CWD: {cwd}")
                            proc.kill()
                            time.sleep(1)
                            killed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not killed_any:
        print(f"[Watcher] No {method} process found for dataset {dataset_path}")


if __name__ == "__main__":
    event_handler = GRNFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=BEELINE_OUTPUT_ROOT, recursive=True)
    observer.start()

    print(f"[Watcher] Monitoring GRN method outputs under {BEELINE_OUTPUT_ROOT}...")

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
