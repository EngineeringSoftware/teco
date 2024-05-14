import dataclasses
import json
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Set, Type, TypeVar

import seutil as su
from tqdm import tqdm

logger = su.log.get_logger(__name__, su.log.INFO)


T = TypeVar("T", bound=dataclasses.dataclass)


def save_dataset(
    save_dir: Path,
    dataset: List[T],
    exist_ok: bool = True,
    append: bool = False,
    only: Optional[List[str]] = None,
    clz: Optional[Type[T]] = None,
):
    """
    Save a dataset to a directory.
    The dataset is a list of dataclass (all data should have the same class).
    An "id" field is compulsory, and should be unique across the entire dataset (i.e., saved into the same directory); this should be checked by the caller.
    Different fields are saved in different jsonl files in the directory.

    :param dataset: the list of data to save.
    :param save_dir: the path to save.
    :param exist_ok: if False, requires that save_dir doesn't exist; otherwise, existing files in save_dir will be modified.
    :param append: if True, append to current saved data (requires exist_ok=True); otherwise, wipes out existing data at save_dir.
    :param only: only save certain fields; the files corresponding to the other fields are not touched; id is always saved.
    :param clz: the class of the data; automatically inferred from the first element of dataset.
    """
    if len(dataset) == 0:
        logger.warning("No data to save")
        return

    if clz is None:
        clz = type(dataset[0])

    if not dataclasses.is_dataclass(clz):
        raise TypeError(f"Only dataclass is supported, but got {clz}")

    fields: Dict[str, dataclasses.Field] = {f.name: f for f in dataclasses.fields(clz)}
    if "id" not in fields or "fully_deserialized" not in fields:
        raise ValueError(
            f"Not a proper data class: {clz}; need to have id and fully_deserialized fields"
        )

    save_dir.mkdir(parents=True, exist_ok=exist_ok)

    su.io.dump(
        save_dir / "id.jsonl",
        [d.id for d in dataset],
        append=append,
    )

    for name, field in fields.items():
        if name == "id" or name == "fully_deserialized":
            continue

        if (
            isinstance(field, dataclasses.Field)
            and field.metadata.get("shared") is True
        ):
            logger.warning(
                f"Seeing an shared field {name}, not sure how to handle it yet"
            )
            continue

        if only is None or name in only:
            su.io.dump(
                save_dir / f"{name}.jsonl",
                [getattr(d, name) for d in dataset],
                append=append,
                serialization=True,
            )


def iter_load_dataset(
    save_dir: Path,
    clz: Type[T],
    only: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    expected_ids: Set[int] = None,
    fully_deserialize: bool = True,
) -> Generator[T, None, None]:
    """
    Iteratively load dataset from the save directory.
    :param save_dir: the directory to load data from.
    :param clz: the class of the data.
    :param only: only load certain fields; the other fields are not filled in the loaded data; id is always loaded.
    :param exclude: do not load certain fields.
    :param expected_ids: if provided, only load data with specified ids.
    :return: a generator iteratively loading the dataset.
    """
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Not found saved data at {save_dir}")

    # figure out the fields to load
    fields: Dict[str, dataclasses.Field] = {f.name: f for f in dataclasses.fields(clz)}
    if "id" not in fields or "fully_deserialized" not in fields:
        raise ValueError(
            f"Not a proper data class: {clz}; need to have id and fully_deserialized fields"
        )
    del fields["id"]
    del fields["fully_deserialized"]

    if only is not None:
        fields = {n: f for n, f in fields.items() if n in only}
    if exclude is not None:
        fields = {n: f for n, f in fields.items() if n not in exclude}

    # open the relevant files
    f2file = {}
    for name, field in fields.items():
        if not (save_dir / f"{name}.jsonl").is_file():
            logger.warning(
                f"Not found {name}.jsonl at {save_dir}, this field won't be loaded"
            )
            continue

        if (
            isinstance(field, dataclasses.Field)
            and field.metadata.get("shared") is True
        ):
            logger.warning(
                f"Seeing an shared field {name}, not sure how to handle it yet"
            )
            continue
        f2file[name] = open(save_dir / f"{name}.jsonl", "r")

    # load all ids
    ids = su.io.load(save_dir / "id.jsonl")

    f2value = {}
    for i in ids:
        if expected_ids is not None and i not in expected_ids:
            # skip this line in all files
            for f in f2file:
                f2file[f].readline()
            continue

        # load fields
        f2value["id"] = i
        for f in f2file:
            f2value[f] = json.loads(f2file[f].readline())
            if fully_deserialize:
                f2value[f] = su.io.deserialize(f2value[f], fields[f].type)

        yield clz(fully_deserialized=fully_deserialize, **f2value)

    # close all files
    for file in f2file.values():
        file.close()


def load_all_dataset(
    save_dir: Path,
    clz: Type[T],
    only: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    fully_deserialize: bool = True,
    pbar: Optional[tqdm] = None,
) -> List[T]:
    """
    Loads the dataset, entirely, from save_dir.
    Does not support only loading partial lines, and does not support line-level progress bar, but faster.

    Other parameters are the same as iter_load_dataset.
    """
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Not found saved data at {save_dir}")

    # figure out the fields to load
    fields: Dict[str, dataclasses.Field] = {f.name: f for f in dataclasses.fields(clz)}
    if "id" not in fields or "fully_deserialized" not in fields:
        raise ValueError(
            f"Not a proper data class: {clz}; need to have id and fully_deserialized fields"
        )
    del fields["id"]
    del fields["fully_deserialized"]

    if only is not None:
        fields = {n: f for n, f in fields.items() if n in only}
    if exclude is not None:
        fields = {n: f for n, f in fields.items() if n not in exclude}

    # load ids
    ids = su.io.load(save_dir / "id.jsonl")

    # load each field
    if pbar is not None:
        pbar.reset(len(fields))

    f2fields = {}
    for name, field in fields.items():
        if pbar is not None:
            pbar.set_description(f"Loading {name}")

        if not (save_dir / f"{name}.jsonl").is_file():
            logger.warning(
                f"Not found {name}.jsonl at {save_dir}, this field won't be loaded"
            )
            continue

        if (
            isinstance(field, dataclasses.Field)
            and field.metadata.get("shared") is True
        ):
            logger.warning(
                f"Seeing an shared field {name}, not sure how to handle it yet"
            )
            continue
        field_type = field.type if fully_deserialize else None
        f2fields[name] = su.io.load(save_dir / f"{name}.jsonl", clz=field_type)

        if pbar is not None:
            pbar.update(1)

    # assemble the dataset
    dataset = [None] * len(ids)
    for i in range(len(ids)):
        dataset[i] = clz(
            id=ids[i],
            fully_deserialized=fully_deserialize,
            **{f: f2fields[f][i] for f in f2fields},
        )

    return dataset


def load_dataset(
    save_dir: Path,
    clz: Type[T],
    only: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    expected_ids: Set[int] = None,
    fully_deserialize: bool = True,
    pbar: Optional[tqdm] = None,
    load_all_then_drop: bool = False,
) -> List[T]:
    """
    Loads the dataset from save_dir.

    :param pbar: an optional progress bar.
    Other parameters are the same as iter_load_dataset.
    """
    if expected_ids is None:
        # use load_all_dataset if no expected ids is needed
        return load_all_dataset(
            save_dir=save_dir,
            clz=clz,
            only=only,
            exclude=exclude,
            fully_deserialize=fully_deserialize,
            pbar=pbar,
        )

    if load_all_then_drop:
        # load all data and drop the ones not in expected_ids
        dataset = load_all_dataset(
            save_dir=save_dir,
            clz=clz,
            only=only,
            exclude=exclude,
            fully_deserialize=fully_deserialize,
            pbar=pbar,
        )
        return [d for d in dataset if d.id in expected_ids]

    dataset = []

    if pbar is not None:
        pbar.set_description("Loading dataset")
        pbar.reset(len(expected_ids))

    for d in iter_load_dataset(
        save_dir=save_dir,
        clz=clz,
        only=only,
        exclude=exclude,
        expected_ids=expected_ids,
        fully_deserialize=fully_deserialize,
    ):
        dataset.append(d)
        if pbar is not None:
            pbar.update(1)

        if len(dataset) == len(expected_ids):
            # Stop early if enough data are loaded
            break

    return dataset
