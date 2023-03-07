from inspect import ArgSpec, getfullargspec
from unittest.mock import patch

import invoke
import seutil as su
from invoke import Collection

from teco.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)
su.log.setup(Macros.log_file)


# ========================================
# fix invoke library not working well with type annotations
# monkeypatch.py


def fix_annotations():
    """
    Pyinvoke doesnt accept annotations by default, this fix that
    Based on: https://github.com/pyinvoke/invoke/pull/606
    """

    def patched_inspect_getargspec(func):
        spec = getfullargspec(func)
        return ArgSpec(*spec[0:4])

    org_task_argspec = invoke.tasks.Task.argspec

    def patched_task_argspec(*args, **kwargs):
        with patch(target="inspect.getargspec", new=patched_inspect_getargspec):
            return org_task_argspec(*args, **kwargs)

    invoke.tasks.Task.argspec = patched_task_argspec


fix_annotations()
# ========================================


from teco.tasks import data, exp

ns = Collection()
ns.add_collection(Collection.from_module(data))
ns.add_collection(Collection.from_module(exp))
