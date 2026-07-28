import os
import csv
import time
import logging
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
import psutil
import numpy as np

# Initialize system/process references once at startup
_process = psutil.Process(os.getpid())
_last_cpu_call_time = time.perf_counter()
_process.cpu_percent(interval=None)
psutil.cpu_percent(interval=None)

# Create stt_logs directory
LOG_DIR = "stt_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate timestamped CSV file name
timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
CSV_LOG_FILE = os.path.join(LOG_DIR, f"app_{timestamp_str}.csv")

# ---------------- DYNAMIC STATS COLLECTOR ----------------
def get_system_stats():
    """
    Collects rich execution stats: Process CPU, System CPU, Process RAM, and System RAM.
    Uses a 100ms measurement window on the first call to guarantee accurate startup CPU readings.
    """
    global _last_cpu_call_time
    now = time.perf_counter()
    elapsed = now - _last_cpu_call_time
    
    # If the time since the last call is too short (e.g., at startup),
    # force a 100ms sampling window to get a real, accurate CPU percentage.
    if elapsed < 0.1:
        proc_cpu = _process.cpu_percent(interval=0.1)
        sys_cpu = psutil.cpu_percent(interval=None)
        _last_cpu_call_time = time.perf_counter()
    else:
        proc_cpu = _process.cpu_percent(interval=None)
        sys_cpu = psutil.cpu_percent(interval=None)
        _last_cpu_call_time = now
        
    # Process RAM RSS in MB
    proc_ram = _process.memory_info().rss / (1024 * 1024)
    
    # System RAM available in GB
    sys_mem = psutil.virtual_memory()
    sys_ram_avail = sys_mem.available / (1024**3)
    
    return {
        "proc_cpu": f"{proc_cpu:.1f}",
        "sys_cpu": f"{sys_cpu:.1f}",
        "proc_ram": f"{proc_ram:.1f}",
        "sys_ram_avail": f"{sys_ram_avail:.2f}"
    }

# ---------------- CUSTOM CSV LOGGING HANDLER ----------------
class CSVLoggingHandler(logging.Handler):
    """Custom logging handler to write detailed system resources and segment metadata into a CSV file."""
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        # Write rich headers if file is new
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", 
                    "LogLevel", 
                    "Process_CPU_Percent", 
                    "System_CPU_Percent", 
                    "Process_RAM_MB", 
                    "System_Available_RAM_GB",
                    "Total_Speakers",
                    "Speaker",
                    "Language",
                    "Segment_Start",
                    "Segment_End",
                    "Message"
                ])

    def emit(self, record):
        try:
            # Collect dynamic execution stats
            stats = get_system_stats()
            timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            log_level = record.levelname
            message = record.getMessage()
            
            # Retrieve optional logging metadata (defaults to empty strings if not provided)
            total_speakers = getattr(record, "total_speakers", "")
            speaker = getattr(record, "speaker", "")
            language = getattr(record, "language", "")
            segment_start = getattr(record, "segment_start", "")
            segment_end = getattr(record, "segment_end", "")
            
            with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, 
                    log_level, 
                    stats["proc_cpu"], 
                    stats["sys_cpu"], 
                    stats["proc_ram"], 
                    stats["sys_ram_avail"],
                    total_speakers,
                    speaker,
                    language,
                    segment_start,
                    segment_end,
                    message
                ])
        except Exception:
            self.handleError(record)

# ---------------- LOGGER CONFIGURATION ----------------
logger = logging.getLogger("speech-to-text")
logger.setLevel(logging.DEBUG)

# File handler for CSV (exclusively handles all logging output)
csv_handler = CSVLoggingHandler(CSV_LOG_FILE)
csv_handler.setLevel(logging.DEBUG)
logger.addHandler(csv_handler)

# Initial system metrics logging (goes strictly into the CSV log)
logger.info("Logging system initialized. Outputting structured logs exclusively to CSV.")
logger.info(f"System details: Python PID={os.getpid()}, Logical CPUs={psutil.cpu_count()}, OS RAM Total={psutil.virtual_memory().total / (1024**3):.1f} GB")

# ---------------- DECORATORS & CONTEXT MANAGERS ----------------
@contextmanager
def log_section(name):
    logger.info(f"=== Starting Section: {name} ===")
    start_time = time.perf_counter()
    start_mem = _process.memory_info().rss / (1024 * 1024)
    try:
        yield
    except Exception as e:
        logger.exception(f"Error occurred in section '{name}': {e}")
        raise
    finally:
        end_time = time.perf_counter()
        end_mem = _process.memory_info().rss / (1024 * 1024)
        logger.info(f"=== Finished Section: {name} (Duration: {end_time - start_time:.3f}s | RAM Change: {end_mem - start_mem:+.1f}MB) ===")

def instrument_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Entering '{func_name}' - Arguments: args={args}, kwargs={kwargs}")
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            
            # Formulate metadata about the output
            out_meta = "None"
            if result is not None:
                if isinstance(result, list):
                    out_meta = f"list of length {len(result)}"
                elif isinstance(result, np.ndarray):
                    out_meta = f"ndarray (shape={result.shape}, dtype={result.dtype})"
                else:
                    out_meta = f"{type(result).__name__}"
            
            logger.debug(f"Exiting '{func_name}' - Output: {out_meta} | Time: {duration:.4f}s")
            return result
        except Exception as e:
            logger.error(f"Error in '{func_name}': {e}", exc_info=True)
            raise
    return wrapper
