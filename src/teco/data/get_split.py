import collections
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import seutil as su
from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw, Path_fr
from seutil.project import Project
from teco.data.data_cls import get_data_cls
from teco.data.utils import load_dataset
from teco.macros import Macros
from tqdm import tqdm

logger = su.log.get_logger(__name__)


class GetSplit:
    def get_split_from_repos(
        self,
        repos_file: Path_fr,
        data_dir: Path_drw,
        out_dir: Union[Path_drw, Path_dc],
        data_cls_name: str = "Data",
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        limit_test: Optional[int] = None,
        seed: int = 7,
    ):
        repos_file = Path(repos_file.abs_path)
        data_dir = Path(data_dir.abs_path)
        out_dir = Path(out_dir.abs_path)
        data_cls = get_data_cls(data_cls_name)
        random.seed(7)

        # load projects
        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        proj2sn: Dict[str, str] = {p.full_name: p.data["sources"] for p in projects}

        # load data
        with tqdm(desc="loading data") as pbar:
            dataset: List[data_cls] = load_dataset(
                data_dir, clz=data_cls, only=["proj_name"], pbar=pbar
            )
        sn2ids: Dict[str, List[str]] = collections.defaultdict(list)
        for d in dataset:
            if d.proj_name not in proj2sn:
                # project removed after process_raw_data
                continue
            sn2ids[proj2sn[d.proj_name]].append(d.id)

        # potentially limit data
        if limit_train is not None:
            random.shuffle(sn2ids[Macros.train])
            sn2ids[Macros.train] = sn2ids[Macros.train][:limit_train]
            sn2ids[Macros.train].sort()
        if limit_val is not None:
            random.shuffle(sn2ids[Macros.val])
            sn2ids[Macros.val] = sn2ids[Macros.val][:limit_val]
            sn2ids[Macros.val].sort()
        if limit_test is not None:
            random.shuffle(sn2ids[Macros.test])
            sn2ids[Macros.test] = sn2ids[Macros.test][:limit_test]
            sn2ids[Macros.test].sort()

        # save split
        su.io.mkdir(out_dir)
        for sn, ids in sn2ids.items():
            print(f"{sn}: {len(ids)}")
            su.io.dump(out_dir / f"{sn}.json", ids)


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(GetSplit, as_positional=False)
