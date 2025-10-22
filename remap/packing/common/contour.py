import numpy as np

class Contour:
    def __init__(self):
        self.terminal = None
        self.contour = None
        self.bottom = None
    
    def initialize(self, xl, xh, platform=0, bottom=True):
        self.terminal = np.array([-1e8, xl, xh, 1e8], dtype=np.float_)
        self.contour = np.array([platform] * 3, dtype=np.float_)
        self.bottom = bottom
        
    @classmethod
    def from_contour(cls, contour: "Contour"):
        _contour = cls()
        _contour.to(contour, copy=True)
        return _contour
    
    def __copy__(self):
        return Contour.from_contour(self)
    
    def to(self, contour: "Contour", *, copy=False):
        self.terminal = contour.terminal.copy() if copy else contour.terminal
        self.contour = contour.contour.copy() if copy else contour.contour
        self.bottom = contour.bottom
        
    def get_platform(self, xl, xh):
        assert xl < xh, "xl >= xh"
        assert xl >= self.terminal[0] and xh <= self.terminal[-1], (xl, xh)
        overlap = ~np.logical_or(
            (self.terminal >= xh)[:-1],
            (self.terminal <= xl)[1:],
        )
        platform = np.max(self.contour[overlap]) if self.bottom else np.min(self.contour[overlap])
        return platform
    
    def add_segment(self, xl, xh, platform):
        lexl = self.terminal <= xl
        gtxh = self.terminal >  xh
        self.terminal = np.concatenate([self.terminal[lexl], np.array([xl, xh]), self.terminal[gtxh]])
        self.contour = np.concatenate([self.contour[lexl[:-1]], np.array([platform]), self.contour[gtxh[1:]]])
        duplicated = np.concatenate([self.terminal[1:] == self.terminal[:-1], np.array([False])])
        self.terminal = self.terminal[~duplicated]
        self.contour = self.contour[~duplicated[:-1]]
        duplicated = np.concatenate([np.array([False]), self.contour[1:] == self.contour[:-1], np.array([False])])
        self.terminal = self.terminal[~duplicated]
        self.contour = self.contour[~duplicated[:-1]]
        assert len(self.terminal) - len(self.contour) == 1
        
    def auc(self, horizon=None):
        horizon = horizon if horizon is not None else (np.min if self.bottom else np.max)(self.contour)
        seglen = self.terminal[1:] - self.terminal[:-1]
        seghgt = self.contour - horizon if self.bottom else horizon - self.contour
        auc = np.sum(np.multiply(seglen, seghgt))
        return auc
    
    def __str__(self):
        content = ""
        for tml, cth in zip(self.terminal[:-1], self.contour):
            content += "({})--{}--".format(tml, cth)
        content += "({})".format(self.terminal[-1])
        return content
    
    def __expr__(self):
        return self.__str__()
