import time
from typing import Callable, Union

from .definition import Singleton

@Singleton
class Timer:
    def __init__(self):
        self._time_table = {}
        self._call_table = {}
        self._hier_table = {}
        self._hierachy = 0
        
    def _initialize(self, name, hierachy):
        self._time_table.setdefault(name, 0)
        self._call_table.setdefault(name, 0)
        self._hier_table.setdefault(name, hierachy)
        
    def _record(self, name, used_time):
        self._time_table[name] += used_time
        self._call_table[name] += 1
        
    def listen(self, name: Union[str, None] = None):
        def listener(func: Callable):
            nonlocal name
            if name is None:
                name = "{}()".format(func.__qualname__)
            def wrapped_func(*args, **kwargs):
                self._initialize(name, self._hierachy)
                start_time = time.time()
                self._hierachy += 1
                ret = func(*args, **kwargs)
                self._hierachy -= 1
                self._record(name, time.time() - start_time)
                return ret
            return wrapped_func
        return listener
    
    def listen_handler(self, name: str):
        while True:
            handler_used = False
            self._initialize(name, self._hierachy)
            self._hierachy += 1
            def handler(start_time):
                nonlocal handler_used
                self._hierachy -= 1
                self._record(name, time.time() - start_time)
                handler_used = True
            start_time = time.time()
            yield lambda: handler(start_time)
            if not handler_used:
                break
            
    def report(self, *, logging_fn=print):
        logging_fn("runtime analysis:")
        for name in self._time_table.keys():
            h = self._hier_table[name]
            t = self._time_table[name]
            c = self._call_table[name]
            if c == 0:
                continue
            logging_fn(
                ("{}{name}: "
                 "called {called_times}, "
                 "takes {total_time:.6f} second(s) in total, "
                 "{average_time:.6f} second(s) on average."
                ).format(
                    "  " * (h + 1),
                    name=name,
                    called_times=c,
                    total_time=t,
                    average_time=t / c,
                )
            )
        
    def export(self) -> dict:
        table = {}
        for name in self._time_table.keys():
            t = self._time_table[name]
            c = self._call_table[name]
            table[name] = {
                "total_time": t,
                "called_times": c,
                "average_time": t / c
            }
        return table
    