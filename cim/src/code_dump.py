# In this file, we use the analysis_parser to generate the python_code with the json.txt

from cim.src import analysis_parser as ap

def generate_code(codegen, result_path, bias_scale_path, concat_list):
    my_model=ap.NetGraph()
    my_model.load_node(codegen, concat_list)
    my_model.analysis_concat_upsample()
    my_model.allocate_mux()
    my_model.generate_to_code(result_path, bias_scale_path)
