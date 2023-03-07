from pathlib import Path
from typing import Optional, Union

from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw, Path_fr

from teco.model.load_trained import load_trained


def save_pretrained(
    model_cls: str,
    ckpt_path: Path_fr,
    output_dir: Optional[Union[Path_drw, Path_dc]] = None,
):
    ckpt_path = Path(ckpt_path.abs_path)
    if output_dir is not None:
        output_dir = Path(output_dir.abs_path)
    else:
        output_dir = ckpt_path.parent.parent / "pretrained"

    load_trained(model_cls, ckpt_path).save_pretrained(output_dir)


if __name__ == "__main__":
    CLI(save_pretrained, as_positional=False)
