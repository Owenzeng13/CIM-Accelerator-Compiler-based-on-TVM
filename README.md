# CIM-Accelerator-Compiler-based-on-TVM
*This is a project named CIM-Accelerator-Compiler-based-on-TVM.*

---
## About the compiler and the files
**The compiler**
1. **TVM-based**

We use TVM because it has an unified IR for all kinds of Network structure input. And in the compiler example, the input definition of the network is described in python with pytorch. Also, TVM has BYOC (Bring your own codegen), which can help us easily fuse the ops into the pattern our device supports and generate a JSON format.

2. **CIM-Accelerator-Compiler**

With the JSON format generated by TVM, we analysis the JSON node and use our algorithm to do the mapping and placement work. Our CIM accelerator supports only a few kinds of op nodes, which have the similar pattern. With the nodes pattern, we analysis the storage of the weights and the activations, then map and place the data on 1 muyan(our CIM accelerator). Finally, we generate an inst file which can work on both FPGA and the real chip with the compiler.

**The files**
1. **bias_scale**

In this file, there are the bias and the scale file with the correct format. And now there are the corresponding files for our example,parts of a modified tiny yolo_v3 model. Also we provide the generate file if you want to generate your own bias/scale file.

2. **cim**

In this file, there are the main algorithm of compiling. `knight` defines the device's inst parser. `model` is for saving the input network model. `src` includes the algorithm.  

3. **result**

The intermediate and final result will be saved in this file, in case you may want to check the proccess. Of course, the final result `test_inst.txt` will also be saved here if you run the `run_case.py`.

4. **tvm**

Here is the modified tvm.

## How to work

First clone the source of our work from github.
```
git clone https://github.com/Owenzeng13/CIM-Accelerator-Compiler-based-on-TVM.git CIM_Compiler_based_TVM
```

1. **Set up with TVM first**

The minimal building requirements for the TVM libraries are : *GCC 7.1*, *Clang 5.0*, *Apple Clang 9.3*, *Cmake 3.18*. And python module: *onnx*, *re*
If you want to use CUDA, CUDA toolkit version >= 8.0 is required. 

Here is a block about how to set up TVM's environment in Ubuntu20.04 in Chinese [How to install TVM (Chinese blog)](https://blog.csdn.net/weixin_42189664/article/details/125842617)

Also it is necessary to build `LLVM` in the environment. The step of installing it is also shown in the blog above. 

Then we will build the modified TVM. It needs some time(maybe several minutes without errors occur)
```
cd CIM_Compiler_based_TVM/tvm/tvm-for-cim
mkdir build
cp cmake/test.cmake build/config.cmake
cd build
cmake ..
make -j4
```

Then, set up the env params.
```
gedit ~/.bashrc
```
Add these sentences in the `.bashrc` file
```
export TVM_HOME=/path/to/tvm-for-cim
export CIM_HOME=/path/to/CIM_Compiler_based_TVM
export PYTHONPATH=$TVM_HOME/python:$CIM_HOME/cim:${PYTHONPATH}
```

In the end, check if it works.
```
python
>> import tvm
>> import cim
>> from tvm.relay.op.contrib import cim
```
If there is no error report, it means you successfully build up the environment.

2. **Complete the info in `run_case.py` and the model**

Edit `run_case.py`. It needs you to fill up the input shape of the tensor, the concat nodes index(only considering the conv node), and the path to the result and bias_scale file(it needs absolute path, not the relative path)

If you want to change the example model(we define a modified Tiny yolo), just open the `cim/model/test_model.py` to edit it following the format shown in that file. Also, don't forget to correct the bias_scale file if you want to modify the conv ops in the model. 

3. **Run the case file**

```
cd CIM_Compiler_based_TVM
python run_case.py
```

4. **To check the result**

The result files will be stored in `CIM_Compiler_based_TVM/result`. 
- **test_model.onnx** : the onnx format of the test_model
- **relayIR.txt** ： the generated relayIRs while compiling with TVM
- **json.txt** : the json format of the model
- **python_code.py** : the python code of the inst
- **test_inst.txt** : the binary inst that can work on FPGA and our CIM device
- **for_test.txt** : it is just a test file