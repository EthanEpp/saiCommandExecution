# src/utils/timing.py
import time
import logging

# Configure logging
logging.basicConfig(filename='timing.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds")
        return result
    return wrapper
