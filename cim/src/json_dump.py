# In this file, we will create an onnx version and a relayIR version of our model.
# Then, we use tvm to help fuse the ops into patterns we support.
# In the end, we will generate a op list of json format, and generate a readable version of json format.
# All the operations are defined as functions.

import torch
import tvm
import onnx
from tvm import relay
from tvm.relay.op.contrib import cim
from tvm.relay.build_module import bind_params_by_name
from cim.model.test_model import TestModel

from itertools import zip_longest, combinations
import json
import numpy as np
from collections import OrderedDict 

def generate_onnx(input_shape, result_path):
    test_model = TestModel()
    onnx_file_path = result_path + "/test_model.onnx"
    input_data = torch.randn(*input_shape)
    torch.onnx.export(model=test_model, args=input_data, f=onnx_file_path, verbose=True, input_names=['input'], output_names=['output'], opset_version=9)


def generate_relay(input_shape, result_path):
    relay_file = result_path + "/relayIR.txt"
    with open(relay_file, "w+") as f:
        # load onnx model and transfer it into relayIR
        onnx_model = onnx.load(result_path + "/test_model.onnx")
        mod, params = relay.frontend.from_onnx(onnx_model,{"input":input_shape})
        
        # bind the params and print the origin relayIR
        if params:
            mod['main'] = bind_params_by_name(mod['main'], params)
        f.write("The origin relayIR:\n")
        f.write(mod["main"].astext(show_meta_data = False))
        f.write("\n============================\n")
        
        target = "cim"
        patterns = cim.get_CIM_pattern()
        
        #apply patterns-based
        mod2 = relay.transform.MergeComposite(patterns)(mod)
        f.write("After patterns-based:\n")
        f.write(mod2["main"].astext(show_meta_data = False))
        f.write("\n============================\n")

        #apply op-based
        mod3 = relay.transform.AnnotateTarget(target)(mod2)
        f.write("After op-based:\n")
        f.write(mod3["main"].astext(show_meta_data = False))
        f.write("\n============================\n")
        
        #graph_partition
        mod4 = relay.transform.PartitionGraph()(mod3)
        f.write("After graph_partition:\n")
        f.write(mod4["main"].astext(show_meta_data = False))
    
    return mod4, params

def generate_json(module, params, result_path):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib = relay.build(module, target=target, params=params)
        
    cim_modules = list(filter(lambda mod: mod.type_key == "cim", lib.get_lib().imported_modules))

    source = []
    codegen = []

    for i in range(len(cim_modules)):
        source.append(cim_modules[i].get_source("json"))
        codegen.append(json.loads(source[i])["nodes"])
    
    json_path = result_path + "/json.txt"
    
    with open(json_path, 'w+') as f:
        f.write(json.dumps(codegen, sort_keys=True, indent=2))
        
    return codegen

def generate_all(input_shape, result_path):
    
    generate_onnx(input_shape, result_path)
    mod, params = generate_relay(input_shape, result_path)
    codegen = generate_json(mod, params, result_path)
    
    return codegen