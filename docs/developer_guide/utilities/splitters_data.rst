.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _data_splitting:

*****************
Dataset Splitters
*****************


OpenFL allows you to specify custom data splits **for simulation runs on a single dataset**.

You may apply data splitters differently depending on the OpenFL workflow that you follow. 


Use Dataset Splitters in your Shard Descriptor
===================================================

Apply one of previously mentioned splitting function on your data to perform a simulation. 

``NumPyDataSplitter`` requires a single ``split`` function. The :code:`split` function returns a list of indices which represents the collaborator-wise indices groups.

This function receives ``data`` - NumPy array required to build the subsets of data indices. It could be the whole dataset, or labels only, or anything else.


.. code-block:: python

    X_train, y_train = ... # train set
    X_valid, y_valid = ... # valid set
    train_splitter = RandomNumPyDataSplitter()
    valid_splitter = RandomNumPyDataSplitter()
    # collaborator_count value is passed to DataLoader constructor
    # shard_num can be evaluated from data_path
    train_idx = train_splitter.split(y_train, collaborator_count)[shard_num]
    valid_idx = valid_splitter.split(y_valid, collaborator_count)[shard_num]
    X_train_shard = X_train[train_idx]
    X_valid_shard = X_valid[valid_idx]

.. note::
    By default, the data is shuffled and split equally. See an `example <https://github.com/intel/openfl/blob/develop/openfl/utilities/data_splitters/numpy.py>`_ of :class:`openfl.utilities.data_splitters.EqualNumPyDataSplitter` for details.
    
