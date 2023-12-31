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
"""Arm Compute Library supported operators."""
import numpy as np
import tvm

from tvm.relay.expr import const
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr
from .register import register_pattern_table


def is_arm_compute_runtime_enabled():
    """Check if the ACL graph runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_arm_compute_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def partition_for_arm_compute_lib(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to Arm Compute Library.

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
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(arm_compute_lib_pattern_table()),
            transform.AnnotateTarget("arm_compute_lib"),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)


@register_pattern_table("arm_compute_lib")
def arm_compute_lib_pattern_table():
    """Get the ACL pattern table."""

    def conv_pattern():
        """Create a convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.pad")(wildcard()) | wildcard()
        pattern = is_op("nn.conv2d")(pattern, is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(is_op("nn.relu"))
        return pattern

    def qnn_conv_pattern():
        """Create a quantized convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.pad")(wildcard()) | wildcard()
        pattern = is_op("qnn.conv2d")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(is_op("nn.relu"))
        pattern = is_op("qnn.requantize")(
            pattern, wildcard(), wildcard(), is_constant(), is_constant()
        )
        return pattern

    def dense_pattern():
        """Create a dense (fully-connected) pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        return pattern

    def qnn_dense_pattern():
        """Create a quantized dense (fully-connected) pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("qnn.dense")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = is_op("qnn.requantize")(
            pattern, wildcard(), wildcard(), is_constant(), is_constant()
        )
        return pattern

    def avg_pool2d_pattern():
        """Creates a pattern that matches either quantized
        avg_pool2d or quantized global_avg_pool2d.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("cast")(wildcard())
        pattern = is_op("nn.avg_pool2d")(pattern) | is_op("nn.global_avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        return pattern

    def l2_pool2d_pattern():
        """Create an l2 pooling pattern from equivalent relay operators.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("power")(wildcard(), is_expr(const(2.0)))
        pattern = is_op("nn.avg_pool2d")(pattern)
        pattern = is_op("sqrt")(pattern)
        return pattern

    def check_conv(extract):
        """Check conv pattern is supported by ACL."""
        call = extract
        while call.op.name != "nn.conv2d":
            call = call.args[0]
        return conv2d(call)

    def check_qnn_conv(extract):
        """Check qnn conv pattern is supported by ACL."""
        if extract.attrs.out_dtype != "uint8":
            return False
        call = extract
        while call.op.name != "qnn.conv2d":
            call = call.args[0]
        return qnn_conv2d(call)

    def check_dense(extract):
        """Check conv pattern is supported by ACL."""
        call = extract
        while call.op.name != "nn.dense":
            call = call.args[0]
        return dense(call)

    def check_qnn_dense(extract):
        """Check qnn conv pattern is supported by ACL."""
        if extract.attrs.out_dtype != "uint8":
            return False
        call = extract
        while call.op.name != "qnn.dense":
            call = call.args[0]
        return qnn_dense(call)

    def check_avg_pool2d(extract):
        """Check average pool2d pattern is supported by ACL."""
        if extract.attrs.dtype != "uint8":
            return False
        pool = extract.args[0]
        if pool.args[0].attrs.dtype != "int32":
            return False
        return avg_pool2d(pool, from_quantized_composite=True)

    def check_l2_pool2d(extract):
        """Check l2 pool2d pattern is supported by ACL."""
        pool = extract.args[0]
        return avg_pool2d(pool)

    return [
        ("arm_compute_lib.conv2d", conv_pattern(), check_conv),
        ("arm_compute_lib.qnn_conv2d", qnn_conv_pattern(), check_qnn_conv),
        ("arm_compute_lib.dense", dense_pattern(), check_dense),
        ("arm_compute_lib.qnn_dense", qnn_dense_pattern(), check_qnn_dense),
        ("arm_compute_lib.qnn_conv2d", qnn_conv_pattern(), check_qnn_conv),
        ("arm_compute_lib.avg_pool2d", avg_pool2d_pattern(), check_avg_pool2d),
        ("arm_compute_lib.l2_pool2d", l2_pool2d_pattern(), check_l2_pool2d),
    ]


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.arm_compute_lib")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("reshape")


@tvm.ir.register_op_attr("nn.conv2d", "target.arm_compute_lib")
def conv2d(expr):
    """Check if the external ACL codegen for conv2d should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.groups != 1:
        return False
    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "float32":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "float32":
        return False
    return True


def qnn_conv2d(expr):
    """Check if the external ACL codegen for qnn.conv2d should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.groups != 1:
        return False
    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "int32" and attrs.out_dtype != "":
        return False
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "uint8":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "uint8":
        return False
    return True


@tvm.ir.register_op_attr("nn.dense", "target.arm_compute_lib")
def dense(expr):
    """Check if the external ACL codegen for dense should be used."""
    attrs, args = expr.attrs, expr.args
    data_typ = args[0].checked_type
    if data_typ.dtype != "float32":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 2 or kernel_typ.dtype != "float32":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    return not require_padding([*args, expr.checked_type])


def qnn_dense(expr):
    """Check if the external ACL codegen for qnn.dense should be used."""
    attrs, args = expr.attrs, expr.args
    data_typ = args[0].checked_type
    if data_typ.dtype != "uint8":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 2 or kernel_typ.dtype != "uint8":
        return False
    if attrs.out_dtype != "int32":
        return False
    return not require_padding([*args, expr.checked_type])


@tvm.ir.register_op_attr("nn.max_pool2d", "target.arm_compute_lib")
def max_pool2d(expr):
    """Check if the external ACL codegen for maxpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    typ = args[0].checked_type
    if typ.dtype != "float32" and typ.dtype != "uint8":
        return False
    return require_padding([*args, expr.checked_type])


def require_padding(inputs):
    """Checks whether supplied data will require padding.
    Most of the operators ACL up to 20.11 uses padded data.
    """

    def _check(shape, dtype):
        """NEON has 128bits/16bytes per vector"""
        if len(shape) == 0:
            return False
        return (shape[-1] * np.dtype(dtype).itemsize) % 16 != 0

    for i in inputs:
        if isinstance(i, (tvm.relay.expr.Var, tvm.relay.expr.Call)):
            if _check(i.checked_type.shape, i.checked_type.dtype):
                return True
        elif isinstance(i, tvm.relay.expr.Constant):
            if _check(i.data.shape, i.data.dtype):
                return True
        elif isinstance(i, tvm.ir.tensor_type.TensorType):
            if _check(i.shape, i.dtype):
                return True
        else:
            raise RuntimeException("Not supported input type: %s" % type(i))
    return False


@tvm.ir.register_op_attr("nn.avg_pool2d", "target.arm_compute_lib")
def avg_pool2d(expr, from_quantized_composite=False):
    """Check if the external ACL codegen for avgpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    typ = args[0].checked_type

    if from_quantized_composite:
        if typ.dtype != "int32":
            return False
    else:
        if typ.dtype not in ["float32"]:
            return False
    if attrs.layout != "NHWC":
        return False

    return not require_padding([*args, expr.checked_type])


@tvm.ir.register_op_attr("nn.global_max_pool2d", "target.arm_compute_lib")
def global_max_pool2d(expr):
    """Check if the external ACL codegen for gloval_maxpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    typ = args[0].checked_type
    if typ.dtype not in ["float32", "uint8"]:
        return False
    if attrs.layout != "NHWC":
        return False
    return not require_padding([*args, expr.checked_type])


@tvm.ir.register_op_attr("nn.global_avg_pool2d", "target.arm_compute_lib")
def global_avg_pool2d(expr):
    """Check if the external ACL codegen for global_avgpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    typ = args[0].checked_type
    if typ.dtype not in ["float32"]:
        return False
    if attrs.layout != "NHWC":
        return False
    return not require_padding([*args, expr.checked_type])


@tvm.ir.register_op_attr("maximum", "target.arm_compute_lib")
def maximum(expr):
    """Check if the external ACL codegen for maximum should be used."""
    args = expr.args
    type_a = args[0].checked_type
    type_b = args[0].checked_type
    return (type_a.dtype == "float32") and (type_b.dtype == "float32")


@tvm.ir.register_op_attr("add", "target.arm_compute_lib")
def add(expr):
    """Check if the external ACL codegen for add should be used."""
    args = expr.args
    for typ in [args[0].checked_type, args[1].checked_type]:
        if typ.dtype != "float32":
            return False

    return True


@tvm.ir.register_op_attr("qnn.add", "target.arm_compute_lib")
def qnn_add(expr):
    """Check if the external ACL codegen for add should be used."""
    args = expr.args
    for typ in [args[0].checked_type, args[1].checked_type]:
        if typ.dtype != "uint8":
            return False

    return True
