from pathlib import Path
from typing import Dict, List, Union

import seutil as su
from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw, Path_fr
from transformers import AutoTokenizer, PreTrainedTokenizer

from teco.macros import Macros
from teco.model.subtokenizer import Subtokenization, Subtokenizer

logger = su.log.get_logger(__name__)


class SubtokenizerBPE(Subtokenizer):

    SPECIAL_TOKENS = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<|UNKNOWN|>",
        "pad_token": "<pad>",
        "cls_token": "<cls>",
        "sep_token": "<sep>",
        "mask_token": "<mask>",
    }

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        self.bos_token: str = self.tokenizer.bos_token
        self.eos_token: str = self.tokenizer.eos_token
        self.unk_token: str = self.tokenizer.unk_token
        self.pad_token: str = self.tokenizer.pad_token
        self.cls_token: str = self.tokenizer.cls_token
        self.sep_token: str = self.tokenizer.sep_token
        self.mask_token: str = self.tokenizer.mask_token
        self.bos_token_id: int = self.tokenizer.bos_token_id
        self.eos_token_id: int = self.tokenizer.eos_token_id
        self.unk_token_id: int = self.tokenizer.unk_token_id
        self.pad_token_id: int = self.tokenizer.pad_token_id
        self.cls_token_id: int = self.tokenizer.cls_token_id
        self.sep_token_id: int = self.tokenizer.sep_token_id
        self.mask_token_id: int = self.tokenizer.mask_token_id

    def toks2subtokenization(self, tokens: List[str]) -> Subtokenization:
        subtoks = []
        t2st = {}
        st2t = {}

        for i, token in enumerate(tokens):
            if self.is_special_token(token):
                st = [token]
            else:
                st = self.tokenizer.tokenize(" " + token)

            t2st[i] = (len(subtoks), len(subtoks) + len(st))
            for j in range(len(st)):
                st2t[len(subtoks) + j] = i
            subtoks += st

        return Subtokenization(subtoks, t2st, st2t)

    def str2esubtoks(self, s: str) -> List[str]:
        return self.tokenizer.tokenize(s)

    def esubtok2id(self, subtoks: Union[List[str], str]) -> Union[List[int], int]:
        return self.tokenizer.convert_tokens_to_ids(subtoks)

    def esubtok2subtok(self, subtoks: Union[List[str], str]) -> Union[List[str], str]:
        if isinstance(subtoks, str):
            return self.tokenizer.convert_tokens_to_string([subtoks])
        else:
            return [self.esubtok2subtok(t) for t in subtoks]

    def id2subtok(self, ids: Union[List[int], int]) -> Union[List[str], str]:
        return self.esubtok2subtok(self.tokenizer.convert_ids_to_tokens(ids))

    def id2esubtok(self, ids: Union[List[int], int]) -> Union[List[str], str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def get_vocab(self, exclude_special_tokens: bool = True) -> Dict[str, int]:
        vocab = {}
        for t, i in self.tokenizer.get_vocab().items():
            if exclude_special_tokens and i in self.tokenizer.all_special_ids:
                continue
            t = self.esubtok2subtok(t)
            # if duplicate token, keep the one with higher tid (likely special token)
            vocab[t] = max(i, vocab.get(t, -1))
        return vocab

    def __len__(self):
        return len(self.tokenizer)


def prepare(
    out_dir: Union[Path_dc, Path],
    base: Union[Path_drw, Path_fr, str] = "Salesforce/codet5-base",
    add_simp_insn_tokens: bool = True,
    add_insn_tokens: bool = True,
) -> PreTrainedTokenizer:
    """
    Prepare a huggingface subtokenizer.
    """
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir.abs_path)
    if isinstance(base, (Path_drw, Path_fr)):
        base = Path(base.abs_path)

    tokenizer = AutoTokenizer.from_pretrained(base)

    # Add all required special tokens
    additional_special_tokens = tokenizer.additional_special_tokens
    if add_simp_insn_tokens:
        additional_special_tokens += Subtokenizer.SIMP_INSN_TOKENS
    if add_insn_tokens:
        additional_special_tokens += Subtokenizer.INSN_TOKENS
    additional_special_tokens = list(sorted(set(additional_special_tokens)))
    logger.info(
        f"Final additional special tokens ({len(additional_special_tokens)}): {additional_special_tokens}"
    )
    tokenizer.add_special_tokens(
        dict(
            additional_special_tokens=additional_special_tokens,
            **SubtokenizerBPE.SPECIAL_TOKENS,
        )
    )

    # Save tokenizer
    su.io.mkdir(out_dir, fresh=True)
    tokenizer.save_pretrained(out_dir)

    return tokenizer


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(prepare, as_positional=False)
