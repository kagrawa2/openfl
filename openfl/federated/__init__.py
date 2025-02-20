# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.federated package."""

import os
from importlib import util

from openfl.federated.data import DataLoader  # NOQA
from openfl.federated.plan import Plan  # NOQA
from openfl.federated.task import TaskRunner  # NOQA

if util.find_spec("keras") is not None:
    from openfl.federated.data import KerasDataLoader
    from openfl.federated.task import KerasTaskRunner
if util.find_spec("torch") is not None:
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    from openfl.federated.data import PyTorchDataLoader
    from openfl.federated.task import PyTorchTaskRunner
if util.find_spec("xgboost") is not None:
    from openfl.federated.data import XGBoostDataLoader
    from openfl.federated.task import XGBoostTaskRunner

__all__ = [
    "Plan",
    "TaskRunner",
    "DataLoader",
]
