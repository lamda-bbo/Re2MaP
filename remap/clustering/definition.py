from numpy import ndarray
from abc import abstractmethod, ABC

class Clusterer(ABC):
    _available: bool = False
    
    @property
    def available(self):
        return self._available
    
    @abstractmethod
    def __call__(self):
        """ once called, all properties become available """
        
    @abstractmethod
    def save(self, file: str): ...
    
    @abstractmethod
    def load(self, file: str): ...
    
    @property
    @abstractmethod
    def clusters(self): ...
    
    @property
    @abstractmethod
    def df_matrix(self) -> ndarray: ...
    