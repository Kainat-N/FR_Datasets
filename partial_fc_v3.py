import math
from typing import Callable
import torch
from torch.nn.functional import linear, normalize


 


class PartialFC_V2(torch.nn.Module):
    """
    Single-GPU version of PartialFC_V2.
    Samples a subset of class centers each iteration to compute margin-based softmax loss.
    """

    _version = 2

    def __init__(self,
                 margin_loss: callable,
                 embedding_size: int,
                 num_classes: int,
                 sample_rate: float = 1.0,
                 fp16: bool = False):
        super(PartialFC_V2, self).__init__()

        self.embedding_size = embedding_size
        self.sample_rate = sample_rate
        self.fp16 = fp16
        self.num_classes = num_classes

        # All classes used since there's no sharding
        self.num_local = num_classes
        self.weight = torch.nn.Parameter(torch.randn(self.num_local, embedding_size) * 0.01)

        # Margin softmax wrapper
        if callable(margin_loss):
            self.margin_softmax = margin_loss
        else:
            raise ValueError("margin_loss must be a callable")

        self.last_batch_size = None
        self.weight_index = None

        self.is_updated = True
        self.init_weight_update = True

        self.dist_cross_entropy = CrossEntropyLossSingleGPU()

    def sample(self, labels, index_positive):
        """
        Sample a subset of class centers each iteration (optional).
        """
        with torch.no_grad():
            positive = torch.unique(labels[index_positive]).sort()[0]

            if self.sample_rate < 1.0:
                num_sample = int(self.num_local * self.sample_rate)
                mask = torch.rand(self.num_local, device=self.weight.device)
                mask[positive] = -1  # ensure positives included
                idx = torch.topk(mask, num_sample)[1].sort()[0]
            else:
                idx = torch.arange(self.num_local, device=self.weight.device)

            self.weight_index = idx
            labels[index_positive] = torch.searchsorted(idx, labels[index_positive])
            return self.weight[idx]
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        batch = embeddings.shape[0]
        if self.last_batch_size is None:
            self.last_batch_size = batch
        elif self.last_batch_size != batch:
            raise RuntimeError(f"Batch size changed: {self.last_batch_size} vs {batch}")

        labels = labels.long().view(-1)

        index_positive = torch.zeros_like(labels, dtype=torch.bool)

        if self.sample_rate < 1.0:
            sampled_weight = self.sample(labels, torch.arange(batch, device=labels.device))
        else:
            sampled_weight = self.weight
            self.weight_index = torch.arange(self.num_local, device=self.weight.device)

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings, dim=1)
            norm_weight = normalize(sampled_weight, dim=1)
            logits = linear(norm_embeddings, norm_weight)

        if self.fp16:
            logits = logits.float()

        logits = logits.clamp(-1, 1)
        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss


class CrossEntropyLossSingleGPU(torch.nn.Module):
    """Standard CrossEntropyLoss replacement."""
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)