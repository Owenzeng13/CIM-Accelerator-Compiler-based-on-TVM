# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""NVDLA supported operators."""
import tvm
from tvm.relay.expr import const
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr
from .register import register_pattern_table

def is_nvdla_runtime_enabled():
    """Check if the NVDLA graph runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_nvdla_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False

def partition_for_nvdla(mod, params):
    """Partition the graph greedily offloading supported
    operators to NVDLA codegen.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        # print(params)
        mod['main'] = bind_params_by_name(mod['main'], params)

    seq = tvm.transform.Sequential([transform.AnnotateTarget("nvdla"),
                                    transform.PartitionGraph()])

    return seq.__call__(mod)


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by NVDLA.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by NVDLA.
    """
    @tvm.ir.register_op_attr(op_name, "target.nvdla")
    def _func_wrapper(attrs):
        return supported

    return _func_wrapper

_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("add")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.batch_flatten")
_register_external_op_helper("nn.bias_add")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("transpose")
_register_external_op_helper("reshape")
_register_external_op_helper("nn.adaptive_avg_pool2d")
