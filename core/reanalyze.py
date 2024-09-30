import ray 
import time

@ray.remote
class BatchWorker_CPU(object):
    def __init__(self):
        pass
    
    def run(self):
        start = False
        while True:
            if not start:
                start = ray.get(self.get_start_signal.remote())
                time.sleep(1)
                continue
        
            