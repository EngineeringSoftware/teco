import torch

from teco.data.structures import Consts
from teco.model.subtokenizer import Subtokenizer


class LossScaler:
    """
    A plugin that scales the loss (or generation probability during inference) of
    generating instructions, by increasing the loss of important instructions and
    lowering the loss of other unimportant instructions.

    Important instructions include:
    - INVOKE* for method invocation
    - GET/PUT* for field usage/definition
    """

    ops_important = {
        Consts.op_invokeinterface,
        Consts.op_invokespecial,
        Consts.op_invokestatic,
        Consts.op_invokevirtual,
        Consts.op_getfield,
        Consts.op_putfield,
        Consts.op_getstatic,
        Consts.op_putstatic,
    }
    ops_unimportant = Consts.ops_all - ops_important

    def __init__(
        self,
        subtokenizer: Subtokenizer,
        scale_important: float = 2.0,
        scale_unimportant: float = 0.2,
        ignore_index: int = -100,
    ):
        self.scale_important = scale_important
        self.scale_unimportant = scale_unimportant
        self.ignore_index = ignore_index

        self.eos_tid = subtokenizer.eos_token_id

        self.op2tid = {}
        for op in Consts.ops_all:
            esubtoks = subtokenizer.toks2esubtoks([op])
            if len(esubtoks) != 1:
                raise RuntimeError(f"op {op} not found in the subtokenizer's vocab")
            self.op2tid[op] = subtokenizer.esubtok2id(esubtoks[0])

        self.tids_important = {self.op2tid[op] for op in self.ops_important}
        self.tids_important.add(self.eos_tid)
        self.tids_unimportant = {self.op2tid[op] for op in self.ops_unimportant}

    def __call__(
        self,
        seq: torch.Tensor,  # long [B, S]
        loss: torch.Tensor,  # float [B, S]
    ) -> torch.Tensor:
        batch_size, seq_len = seq.shape
        loss = loss.clone()

        # scan over the sequence and adjust the loss
        for i in range(batch_size):
            important = False
            for j in range(seq_len):
                if seq[i][j].item() in self.tids_important:
                    important = True
                elif seq[i][j].item() in self.tids_unimportant:
                    important = False
                elif seq[i][j].item() == self.ignore_index:
                    continue

                if important:
                    loss[i, j] = loss[i, j] * self.scale_important
                else:
                    loss[i, j] = loss[i, j] * self.scale_unimportant

        return loss
