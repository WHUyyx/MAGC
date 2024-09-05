from typing import Any, overload
from torch.nn import functional as F

class BatchTransform:
    @overload
    def __call__(self, batch: Any) -> Any:
        ...

class IdentityBatchTransform(BatchTransform):
    def __call__(self, batch: Any) -> Any:
        return batch