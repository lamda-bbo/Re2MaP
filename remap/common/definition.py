from abc import abstractmethod
from typing import Generic, TypeVar, Type, Any, Dict, Callable

T = TypeVar('T')

class Factory(Generic[T]):
    @abstractmethod
    def __call__(self) -> T: ...
    
class Singleton(Generic[T]):
    def __init__(self, cls: Type[T]):
        self._cls = cls
        self._instance = None
        
    def __call__(self, *args, **kwargs) -> T:
        if self._instance is None:
            self._instance = self._cls(*args, **kwargs)
        return self._instance
    
class FlyweightPool(object):
    def __init__(self, **kwargs):
        self._pool: Dict[str, Any] = kwargs
        
    def __getattribute__(self, name: str):
        if not name.startswith("_") and name in self._pool:
            return self._pool[name]
        return super().__getattribute__(name)
    
M = TypeVar('M', bound=Callable[[FlyweightPool], T])

def flyweight(meth: M) -> M:
    if meth.__name__.startswith("_"):
        return meth
    def wrapped(self: FlyweightPool):
        ret = meth(self)
        self._pool[__name__] = ret
        return ret
    return wrapped