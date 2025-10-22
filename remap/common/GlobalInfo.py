import os
from typing import Literal
from datetime import datetime
import random

import numpy as np
import torch

from .definition import Singleton
from .timer import Timer

@Singleton
class GlobalInfo:
    _placedb_backend: Literal['DREAMPlace.PlaceDB']
    
    def __init__(self):
        self._placedb_backend = 'DREAMPlace.PlaceDB'
        self._begin_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    @property
    def timer(self):
        return Timer()
    
    @property
    def placedb_backend(self):
        assert self._placedb_backend in \
            ['DREAMPlace.PlaceDB'], \
            "invalid placedb backend: " \
            f"{self._placedb_backend}"
        return self._placedb_backend
    
    _workspace: str
    
    @property
    def workspace(self):
        return self._workspace
    
    @workspace.setter
    def workspace(self, path: str):
        self._workspace = path
        
    _design_name: str
    
    @property
    def design_name(self):
        return self._design_name
    
    @design_name.setter
    def design_name(self, value):
        self._design_name = value
        
    _run_tag: str
    
    @property
    def run_tag(self):
        return self._run_tag
    
    @run_tag.setter
    def run_tag(self, value):
        self._run_tag = value
        
    @property
    def def_savedir(self):
        return os.path.join(self.workspace, self.run_tag, self.begin_time)
    
    @property
    def fig_savedir(self):
        return os.path.join(self.workspace, self.run_tag, self.begin_time)
    
    _begin_time: str
    
    @property
    def begin_time(self):
        return self._begin_time
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, value):
        self._seed = value
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
    
def get_global(attr):
    return getattr(GlobalInfo(), attr, None)

g = GlobalInfo()