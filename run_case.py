# Run this file in Python to get the Compile result in the `result` file
from cim.src import json_dump as jd
from cim.src import code_dump as cd

def main():
    
    #------------------------------------------------- fill information -------------------------------------------------#
    
    # put the input tensor shape here
    input_shape = (1,64,16,32)
    
    # define paths of `the result file` and `the bias_scale file`
    # please use the `absolute path`, NOT the `relative path` 
    result_path = "/home/zengym/Desktop/CIM_Compiler_based_TVM/result"
    bias_scale_path = "/home/zengym/Desktop/CIM_Compiler_based_TVM/bias_scale"
    
    # fill the index of the concat conv_node (just consider the fused-conv node, not includes the concat and upsample node)
    concat_list = [1,4]
    
    #------------------------------------------------- fill information -------------------------------------------------#
    
    codegen = jd.generate_all(input_shape, result_path)
    cd.generate_code(codegen, result_path, bias_scale_path, concat_list)
    python_code_path = result_path + "/python_code.py"
    with open(python_code_path, "r") as f:
        code = compile(f.read(), python_code_path, 'exec')
    try:
        exec(code, globals())
    except Exception as e:
        print(f"An error occurred:{e}")

if __name__ == "__main__":
	main()