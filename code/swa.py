import torch
import operator
import numbers
from pathlib import Path
import cytoolz.curried as ct


class OpsDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __op__(self, other, op):
        if isinstance(other, dict):
            return OpsDict({ k: op(v, other[k]) for k, v in self.items() })
        if isinstance(other, (numbers.Number, torch.Tensor)):
            return OpsDict({ k: op(v, other) for k, v in self.items() })
        
    def __add__(self, other):
        return self.__op__(other, op=operator.add)
    
    def __mul__(self, other):
        return self.__op__(other, op=operator.mul)
    
    def __sub__(self, other):
        return self.__op__(other, op=operator.sub)
    
    def __mod__(self, other):
        return self.__op__(other, op=operator.mod)
    
    def __truediv__(self, other):
        return self.__op__(other, op=operator.truediv)
    
    def __lt__(self, other):
        return self.__op__(other, op=operator.lt)
    
    def __le__(self, other):
        return self.__op__(other, op=operator.le)
    
    def __gt__(self, other):
        return self.__op__(other, op=operator.gt)
    
    def __ge__(self, other):
        return self.__op__(other, op=operator.ge)
    
class SWA(object):
    def __init__(self):
        super().__init__()
        self.wswa = None
        self.nmodels = 0
    
    def add_model(self, model):
        if self.nmodels == 0:
            self.wswa = OpsDict(model)
            self.nmodels = 1
            return
        self.wswa = (self.wswa * self.nmodels + model) / (self.nmodels + 1)
        self.nmodels += 1

@ct.curry
def last_n_swa(n, end_epoch, checkpoints_path, device=None):
    base = Path(checkpoints_path)
    epochs = range(end_epoch-n+1, end_epoch+1)
    swa = SWA()
    for epoch in epochs:
        chk = torch.load(base / f'{epoch}.torch', map_location=device)
        swa.add_model(chk['model'])
    chk['model'] = dict(swa.wswa)
    chk['swa_nmodels'] = swa.nmodels
    return chk
