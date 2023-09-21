# In this file, we analysis the compute graph and do mapping and placement.

import numpy
import math
import re

class GlbBufBank():
    # only for ConvNode
    # global Buffer Bank : 0 for not choose, 1 for rd, 2 for wr, 3 for wr_upsample
    # the order is from 1 to 16 , which is opposite to the list (index+1) order

    def __init__(self):
        self.bank = [0]*16

    def set_rd_mux(self, rd_src, use_num): 
        """set the read banks

        Args:
            rd_src (int): the order(not index, from 1 to 16) of the first read bank
            use_num (int): the numbers of banks needed
        """
        assert rd_src + use_num <= 16, "rd_mux out of range"
        for i in range(0, use_num):
            self.bank[rd_src - 1 + i] = 1

    def set_wr_mux(self, wr_src, use_num):
        """set the write banks

        Args:
            wr_src (int): the order(not index, from 1 to 16) of the first write bank
            use_num (int): the numbers of banks needed
        """
        assert wr_src + use_num <= 16, "wr_mux out of range"
        for i in range(0, use_num):
            self.bank[wr_src - 1 + i] = 2

    def set_upsample_wr_mux(self, upsample_wr_src, use_num):
        """set the upsample write banks

        Args:
            upsample_wr_src (int): the order(not index, from 1 to 16) of the first write bank
            use_num (int): the numbers of banks needed
        """
        assert upsample_wr_src + use_num <= 16, "up_wr_mux out of range"
        for i in range(0, use_num):
            self.bank[upsample_wr_src - 1 + i] = 3

    def get_last_rd_bank(self):
        """get the last read bank's order(not index)

        Returns:
            int: the order
        """ 
        flag = 0
        for i in range(16):
            if self.bank[i] == 1 and i < 15:
                flag = 1

            elif self.bank[i] != 1 and flag == 1 :
                return i
            
            elif i == 15 and self.bank[i] == 1:
                return 16
    
    def get_first_rd_bank(self):
        """get the first read bank's order(not index)

        Returns:
            int: the order
        """
        for i in range(16):
            if self.bank[i] == 1:
                return i + 1
            
    def get_first_wr_bank(self):
        """get the first write bank's order(not index)

        Returns:
            int: the order
        """
        for i in range(16):
            if self.bank[i] == 2:
                return i + 1
            
    def get_first_upsample_bank(self):
        """get the first write bank's order(not index)

        Returns:
            int: the order
        """
        for i in range(16):
            if self.bank[i] == 3:
                return i + 1
    
    
class ConvNode:
    # set a ConvNode
    def __init__(self, node):
        # the param to set in inst
        self.src_base = 0x0 
        self.src_width = 0x0 
        self.src_height = 0x0 
        self.src_channel_16x = 0x0 
        self.cim_base = 0x0 
        self.dst_base = 0x0 
        self.dst_width = 0x0 
        self.dst_height = 0x0 
        self.post_scale_id = -1 
        self.post_bias_id = -1 
        self.post_shift = 0xC # this param is designed by our quan algorithm 
        self.stride = 1 
        self.kernel_size = 0 
        self.upsample = 0 
        self.padding = 0 
        self.relu = -1 
        self.max_pool = -1 
        
        # supportive param
        self.id = -1
        self.type = "conv"
        self.des_channel = 0 # the output channel
        self.set_times = 0 # the times of set param (for 32 output channels per time)
        self.input_graph_size = 0 # the input graph size weight*height*in_channels
        self.output_graph_stride = 0 # the offset of the output address (per param_set)
        self.cim_amount = 0 # the amount of the weights (per param_set)
        self.cim_stride = 0 # the offset of the weight address (per param_set) 
        self.output_num = 1 # for concat and upsample 
        
        #self.biasLine = []  
        #self.scaleLine = []
        
        
        self.rd_start_bank = 0
        self.rd_src_offset = 0
        self.rd_use_num = 0 # number of banks only for read
        self.wr_start_bank = 0
        self.wr_src_offset = 0
        self.wr_use_num = 0 # number of banks only for write
        
        self.concat_start_bank = 0 # only for output_num == 2
        self.concat_offset = 0 # only for output_num == 2
        
        self.GlbBufBank = GlbBufBank() # for each node's buffer mux, choose in graph analysis

        self.set_param_string = [] # the string of set_calc_param
        self.set_GlbBufBank_string = [] # the string of set_GlbBufBank
        
        self.set_upsample_param_string = []

        for part in node:
            if part["op"] == "input":
                self.src_channel_16x = int(math.ceil(part["attrs"]["shape"][0][0][1]/16)) - 1
                self.src_height = part["attrs"]["shape"][0][0][2]
                self.src_width = part["attrs"]["shape"][0][0][3]
            
            elif part["op"] == "kernel":
                if part["name"] == "conv2d_relu_maxpool":
                    self.relu = 1
                    self.max_pool = 1
                elif part["name"] == "conv2d_relu":
                    self.relu = 1
                    self.max_pool = 0
                elif part["name"] == "conv2d_maxpool":
                    self.relu = 0
                    self.max_pool = 1
                elif part["name"] == "conv2d":
                    self.relu = 0
                    self.max_pool = 0

                self.des_channel = part["attrs"]["shape"][0][0][1]
                self.dst_height = part["attrs"]["shape"][0][0][2]
                self.dst_width = part["attrs"]["shape"][0][0][3]

                self.stride = part["attrs"]["strides"][0][0]
                
                self.kernel_size = int(part["attrs"]["kernel_size"][0][1])

        if self.kernel_size == 1:
            self.padding = 0x0
        
        elif self.kernel_size == 3:
            self.padding = 0xF

    '''
    def read_param_line(self, filepath_bias, filepath_scale):
        with open(filepath_bias,'r')as fi:
            biasLine = fi.readlines()
            for i in range(len(biasLine)):
                biasLine[i] = int(biasLine[i],2)
                self.biasLine0.append(biasLine[i])
        fi.close()

        with open(filepath_scale,'r')as fi:
            scaleLine = fi.readlines()
            for i in range(len(scaleLine)):
                scaleLine[i] = int(scaleLine[i],2)
                self.scaleLine0.append(scaleLine[i])
        fi.close()
    '''

    def analysis_node(self):
        """
            to analysis the node and set some other params
        """
        self.set_times = int(self.des_channel/32) # group of set_param
        self.input_graph_size = self.src_width*self.src_height*(self.src_channel_16x + 1)*16 
        self.output_graph_stride = self.dst_height*self.dst_width # per 32 channels
        self.cim_amount = (self.src_channel_16x + 1)*16
        if self.kernel_size == 3:
            self.cim_stride = int(self.cim_amount / 16)
        elif self.kernel_size == 1:
            self.cim_stride = int(self.cim_amount / 128)
            
        self.rd_use_num = int(math.ceil(self.input_graph_size / (256 * 4 * 32)))
        self.wr_use_num = int(math.ceil(self.output_graph_stride * self.set_times / (256 * 4)))
        self.output_graph_upsample_stride = self.output_graph_stride * 4
        self.upsample_use_num = int(math.ceil(self.output_graph_stride * self.set_times * 4 / (256 * 4))) # just consider the upsample stride is (2, 2)
    
    def set_GlbBufBank_param(self):
        rd_mux = 0
        wr_mux = 0
        for i in range(len(self.GlbBufBank.bank)):
            if self.GlbBufBank.bank[i] == 1:
                rd_mux = rd_mux + (1 << i)
            elif self.GlbBufBank.bank[i] == 2:
                wr_mux = wr_mux + (1 << i)
        string = "\tol.npu.set_config_GlbBufBankMux(wr_mux={0}, rd_mux={1})\n".format(wr_mux,rd_mux)
        self.set_GlbBufBank_string.append(string)

    def set_param(self):
        if self.upsample == 0:
            self.src_base = 1024 * (self.rd_start_bank - 1) + self.rd_src_offset
            self.dst_base = 1024 * (self.wr_start_bank - 1) + self.wr_src_offset
            if self.kernel_size == 1:
                self.src_base = int(self.src_base/4)
            for i in range(self.set_times):
                string = "\tol.npu.set_calc_param(src_base={0}, src_width={1}, src_height={2}, src_channel_16x={3}, cim_base={4}, dst_base={5}, dst_width={6}, dst_height={7}, post_scale=scaleLine{19}[{8}:{9}], post_bias=biasLine{19}[{10}:{11}], upsample={12}, post_shift={13}, stride={14}, kernel_size={15}, padding={16}, relu={17}, max_pool={18})\n".format(self.src_base, \
                self.src_width, self.src_height, self.src_channel_16x, self.cim_base + self.cim_stride*i, self.dst_base + self.output_graph_stride * i, self.dst_width, self.dst_height, 32*i , 32*(i+1), 32*i , 32*(i+1), \
                    self.upsample, self.post_shift, self.stride, self.kernel_size, self.padding, self.relu, self.max_pool, self.id)

                self.set_param_string.append(string)

        else: # need to upsample
            if self.output_num == 1:
                self.src_base = 1024 * (self.rd_start_bank - 1) + self.rd_src_offset
                self.dst_base = 1024 * (self.wr_start_bank - 1) + self.wr_src_offset
                # self.dst_height *= 2
                # self.dst_width *= 2
                if self.kernel_size == 1:
                    self.src_base = int(self.src_base/4)
                for i in range(self.set_times):
                    for m in [0, 1, self.dst_width * 2, self.dst_width * 2 + 1]:  
                        string = "\tol.npu.set_calc_param(src_base={0}, src_width={1}, src_height={2}, src_channel_16x={3}, cim_base={4}, dst_base={5}, dst_width={6}, dst_height={7}, post_scale=scaleLine{19}[{8}:{9}], post_bias=biasLine{19}[{10}:{11}], upsample={12}, post_shift={13}, stride={14}, kernel_size={15}, padding={16}, relu={17}, max_pool={18})\n".format(self.src_base, \
                        self.src_width, self.src_height, self.src_channel_16x, self.cim_base + self.cim_stride*i, self.dst_base + self.output_graph_upsample_stride * i + m, self.dst_width, self.dst_height, 32*i , 32*(i+1), 32*i , 32*(i+1), \
                        self.upsample, self.post_shift, self.stride, self.kernel_size, self.padding, self.relu, self.max_pool, self.id)

                        self.set_param_string.append(string)
                
            else:
                # origin output
                self.src_base = 1024 * (self.rd_start_bank - 1) + self.rd_src_offset
                self.dst_base = 1024 * (self.wr_start_bank - 1) + self.wr_src_offset
                self.upsample = 0
                if self.kernel_size == 1:
                    self.src_base = int(self.src_base/4)
                for a in range(self.set_times):
                    string = "\tol.npu.set_calc_param(src_base={0}, src_width={1}, src_height={2}, src_channel_16x={3}, cim_base={4}, dst_base={5}, dst_width={6}, dst_height={7}, post_scale=scaleLine{19}[{8}:{9}], post_bias=biasLine{19}[{10}:{11}], upsample={12}, post_shift={13}, stride={14}, kernel_size={15}, padding={16}, relu={17}, max_pool={18})\n".format(self.src_base, \
                    self.src_width, self.src_height, self.src_channel_16x, self.cim_base + self.cim_stride*a, self.dst_base + self.output_graph_stride * a, self.dst_width, self.dst_height, 32*a , 32*(a+1), 32*a , 32*(a+1), \
                    self.upsample, self.post_shift, self.stride, self.kernel_size, self.padding, self.relu, self.max_pool, self.id)

                    self.set_param_string.append(string)

                # upsample for concat
                self.upsample = 1
                for b in range(self.set_times):
                    # self.dst_height *= 2
                    # self.dst_width *= 2 
                    for m in [0, 1, self.dst_width * 2, self.dst_width * 2 + 1]:  
                        string = "\tol.npu.set_calc_param(src_base={0}, src_width={1}, src_height={2}, src_channel_16x={3}, cim_base={4}, dst_base={5}, dst_width={6}, dst_height={7}, post_scale=scaleLine{19}[{8}:{9}], post_bias=biasLine{19}[{10}:{11}], upsample={12}, post_shift={13}, stride={14}, kernel_size={15}, padding={16}, relu={17}, max_pool={18})\n".format(self.src_base, \
                        self.src_width, self.src_height, self.src_channel_16x, self.cim_base + self.cim_stride * b, 1024 * (self.concat_start_bank - 1) + self.concat_offset + self.output_graph_upsample_stride * b + m, self.dst_width, self.dst_height, 32*b , 32*(b+1), 32*b , 32*(b+1), \
                        self.upsample, self.post_shift, self.stride, self.kernel_size, self.padding, self.relu, self.max_pool, self.id)

                        self.set_upsample_param_string.append(string)
            

class ConcatNode:
    def __init__(self):
        self.concat_node_id = []
        self.type = "concat"
        self.src = 0
        self.banks_num = 0
        self.concat_nearby = False
        self.concat_offset = []
        
class PreConCatNode:
    def __init__(self):
        self.concat_index = -1
        self.type = "pre_concat"
        
class UpsampleNode:
    def __init__(self):
        self.upsample_node_id = []
        self.type = "upsample"

class GraphBufBank:
    def __init__(self):
        self.bank = [0]*16 # 0 for empty, 1 for read, 2 for concat
    
    def set_concat(self, src, num):
        for i in range(num):
            self.bank[src - 1 + i] = 2
    
    def find_empty_part(self, num): 
        flag = 0
        i = 0
        while i + num < 16:
            for index in range(num):
                if self.bank[i + index] == 0:
                    flag += 1
                else:
                    flag = 0
                    break
            if flag == num:
                return i + 1
            else:
                i += 1
        assert 1 == 0, "no empty space"
        
    def set_new_read(self, src, num):
        for i in range(16):
            if self.bank[i] == 1:
                self.bank[i] = 0
        for i in range(num):
            self.bank[src - 1 + i] = 1
            
    def clear_concat(self):
        for i in range(16):
            if self.bank[i] == 2:
                self.bank[i] = 0
    
    def clear_read(self):
        for i in range(16):
            if self.bank[i] == 1:
                self.bank[i] = 0
                
class NetGraph:
    def __init__(self):
        self.node = []
        self.src_base = 0x0
        self.dst_base = 0x0
        self.cim_base = 0x0
        self.conv_node = []

        # supportive param
        self.concat_index = -1
        self.upsample_index = -1
        self.GraphBufBank = GraphBufBank() # for helping mux allocate
        
    def load_node(self, module_list, concat_list):
        """To load the nodes from module_list

        Args:
            module_list (list): the node list to be loaded
            concat_list (list): the concat node index in the graph, for moving 
                                the pre_concat nodes right after them
        """
        self.node = [0] * len(module_list)
        for module in module_list:
            flag = 0 # for whether there is kernel part
            for part in module:
                if part["op"] == "input":
                    name = str(part["name"])
                    index = int(re.findall('\d+', name)[0])
                if part["op"] == "kernel":
                    if part["name"] == "concatenate":
                        node = ConcatNode()
                    elif part["name"] == "nn.upsampling":
                        node = UpsampleNode()
                    else:
                        node = ConvNode(module)
                        node.analysis_node()
                    flag = 1
            if flag == 1:
                self.node[index] = node
                
        calc_node = []
        for index in range(len(module_list)):
            if self.node[index] != 0: # it is a node
                calc_node.append(self.node[index])
        
        while len(concat_list) > 0 :
            index = concat_list.pop()
            calc_node.insert(index + 1, PreConCatNode())
        
        for i in range(len(self.node)):
            self.node[i] = calc_node[i]
        
        for i in self.node:
            if i.type == "conv":
                self.conv_node.append(self.node.index(i))
                i.id = self.conv_node.index(self.node.index(i))

    def analysis_concat_upsample(self):
        # only for 1 photo, so now dont need to change the dst_base and the src_base, just make sure
        # the rd_mux and the wr_mux is correct, and set the right cim_base
        
        # search the node list to build up concat and upsample infomation, now 
        # only support "1 concat and 1 upsample, and the concat must be before upsample"
        # and the concat needs to have just 2 nodes to concat
        
        # find the concat node first
        index = len(self.node) - 1 # from back to the front
        while index >= 0:
            if self.node[index].type == "concat":
                self.concat_index = index
                while index >= 0:
                    if self.node[index].type == "pre_concat":
                        self.node[index].concat_index = self.concat_index
                        if len(self.node[self.concat_index].concat_node_id) != 0:
                            pre_concat_node = self.node[self.concat_index].concat_node_id.pop()
                            if pre_concat_node - index == 1:
                                self.node[self.concat_index] = True
                            self.node[self.concat_index].concat_node_id.append(pre_concat_node)
                        self.node[self.concat_index].concat_node_id.append(index - 1)
                        index -= 2
                    else: 
                        index -= 1
                break
            else:
                index -= 1
        
        # find the upsample node
        index = len(self.node) - 1 # from back to the front
        while index >= 0:
            if self.node[index].type == "upsample":
                self.upsample_index = index
                index -= 1
                if self.node[index].type == "conv":
                    self.node[self.upsample_index].upsample_node_id.append(index)
                    self.node[index].upsample = 1
                    break
                
                elif self.node[index].type == "concat":
                    for i in self.node[index].concat_node_id:
                        self.node[self.upsample_index].upsample_node_id.append(i)
                        if i == index - 2: # the conv-node near concat node
                            self.node[i].upsample = 1
                        
                        else:
                            self.node[i].upsample = 1
                            self.node[i].output_num = 2
                break
            else: 
                index -= 1
        
        # then we need to calculate the bank_num of concat
        # handle the concat node
        # offset is relative to the src_start_banks
        if self.concat_index > -1:
            for i in self.node[self.concat_index].concat_node_id:
                if self.node[i].upsample == 0:
                    self.node[self.concat_index].concat_offset.append(self.node[i].set_times * self.node[i].output_graph_stride)
                    self.node[self.concat_index].banks_num += self.node[i].wr_use_num
            
                else:
                    self.node[self.concat_index].concat_offset.append(self.node[i].set_times * self.node[i].output_graph_stride * 4)
                    self.node[self.concat_index].banks_num += self.node[i].upsample_use_num
                
            if self.node[self.concat_index].concat_nearby == False:
                real_banks_num = math.ceil(sum(self.node[self.concat_index].concat_offset) / 1024)
                self.node[self.concat_index].banks_num = real_banks_num
                self.node[self.concat_index].concat_offset[1] = self.node[self.concat_index].concat_offset[0]
                self.node[self.concat_index].concat_offset[0] = 0 
            
            else:
                first_concat = self.node[self.concat_index].concat_offset[0]
                self.node[self.concat_index].concat_offset[0] = 1024 - (first_concat % 1024)
                second_concat = math.ceil(first_concat / 1024) * 1024
                self.node[self.concat_index].concat_offset[1] = second_concat
            

    def allocate_mux(self):
        # this part is for mux choosing. We analysis the bank_use of each nodes
        # especially the conv nodes and the concat nodes. We finds every node in 
        # these 2 types, and set their rd_src banks and wr_src banks. Attention,
        # the concat nodes just have wr_src banks for they don't need to calculate,
        # just delivery the wr_src banks information.
        
        start_rd_buf = 1
        bank_amount = 16
        concat_flag = 0
        for index in range(len(self.node)):
            if self.node[index].type == "conv":
                
                # set rd_mux
                if index == 0:
                    self.node[index].GlbBufBank.set_rd_mux(start_rd_buf, self.node[index].rd_use_num)
                    self.GraphBufBank.set_new_read(start_rd_buf, self.node[index].rd_use_num)
                    self.node[index].rd_start_bank = start_rd_buf
                else:
                    if self.node[index - 1].type == "conv":
                        last_node = self.node[index - 1]                 
                        self.node[index].rd_start_bank = last_node.wr_start_bank
                        self.node[index].GlbBufBank.set_rd_mux(self.node[index].rd_start_bank, self.node[index].rd_use_num)
                    
                    elif self.node[index - 1].type == "pre_concat":
                        last_node = self.node[index - 2]
                        self.node[index].rd_start_bank = last_node.wr_start_bank
                        self.node[index].rd_src_offset = last_node.wr_src_offset
                        self.node[index].GlbBufBank.set_rd_mux(self.node[index].rd_start_bank, self.node[index].rd_use_num)
                        
                    elif self.node[index - 1].type == "concat":
                        last_node = self.node[index - 1] # concat node
                        self.node[index].rd_start_bank = last_node.src
                        self.node[index].GlbBufBank.set_rd_mux(last_node.src, last_node.banks_num)
                        self.node[index].rd_src_offset = last_node.concat_offset[0]
                            
                    elif self.node[index - 1].type == "upsample":
                        last_node = self.node[index - 2]
                        if last_node.type == "conv":
                            self.node[index].rd_start_bank = last_node.wr_start_bank
                            self.node[index].GlbBufBank.set_rd_mux(self.node[index].rd_start_bank, self.node[index].rd_use_num)
                        elif last_node.type == "concat":
                            self.node[index].rd_start_bank = last_node.src
                            self.node[index].GlbBufBank.set_rd_mux(last_node.src, last_node.banks_num)
                            self.node[index].rd_src_offset = last_node.concat_offset[0]
                    
                # set wr_mux
                if self.node[index].upsample == 0:
                    if self.concat_index > -1: # there is a concat node
                        if index in self.node[self.concat_index].concat_node_id: # this node is a concat node
                            if concat_flag == 0:
                                concat_flag = 1
                                self.node[self.concat_index].src = self.GraphBufBank.find_empty_part(self.node[self.concat_index].banks_num)
                                self.GraphBufBank.set_concat(self.node[self.concat_index].src, self.node[self.concat_index].banks_num)
                            order = self.node[self.concat_index].concat_node_id.index(index)
                            self.node[index].wr_start_bank = self.node[self.concat_index].src
                            self.node[index].wr_src_offset = self.node[self.concat_index].concat_offset[order]
                            self.node[index].GlbBufBank.set_wr_mux(self.node[index].wr_start_bank, self.node[self.concat_index].banks_num) 
                            self.GraphBufBank.clear_read()
                        
                        else:
                            self.node[index].wr_start_bank = self.GraphBufBank.find_empty_part(self.node[index].wr_use_num)
                            self.node[index].GlbBufBank.set_wr_mux(self.node[index].wr_start_bank, self.node[index].wr_use_num)
                            self.GraphBufBank.set_new_read(self.node[index].wr_start_bank, self.node[index].wr_use_num)
                            if index > self.concat_index:
                                self.GraphBufBank.clear_concat()
                            # for concat is ended, no used anymore
                    
                    else:
                        self.node[index].wr_start_bank = self.GraphBufBank.find_empty_part(self.node[index].wr_use_num)
                        self.node[index].GlbBufBank.set_wr_mux(self.node[index].wr_start_bank, self.node[index].wr_use_num)
                        self.GraphBufBank.set_new_read(self.node[index].wr_start_bank, self.node[index].wr_use_num)
                        
                elif self.node[index].upsample == 1:
                    if self.concat_index > -1: # there is a concat node
                        if index in self.node[self.concat_index].concat_node_id:
                            # two cases: the far one and the nearby one 
                            # we have the param: output_num to distinguish them
                            if self.node[index].output_num == 2:
                                # the far one
                                # it means that it has two wr_mux
                                # concat_banks for the upsample result and wr_banks for the origin one
                                
                                # the upsample result
                                if concat_flag == 0:
                                    concat_flag = 1
                                    self.node[self.concat_index].src = self.GraphBufBank.find_empty_part(self.node[self.concat_index].banks_num)
                                    self.GraphBufBank.set_concat(self.node[self.concat_index].src, self.node[self.concat_index].banks_num)
                                order = self.node[self.concat_index].concat_node_id.index(index)
                                self.node[index].concat_start_bank = self.node[self.concat_index].src
                                self.node[index].concat_offset = self.node[self.concat_index].concat_offset[order]
                                self.node[index].GlbBufBank.set_wr_mux(self.node[index].concat_start_bank, self.node[self.concat_index].banks_num) 
                            

                                # the origin result - dont need to upsample
                                self.node[index].wr_start_bank = self.GraphBufBank.find_empty_part(self.node[index].wr_use_num)
                                self.node[index].GlbBufBank.set_wr_mux(self.node[index].wr_start_bank, self.node[index].wr_use_num)
                                self.GraphBufBank.set_new_read(self.node[index].wr_start_bank, self.node[index].wr_use_num)
                                
                            else: # the nearby one
                                order = self.node[self.concat_index].concat_node_id.index(index)
                                self.node[index].wr_start_bank = self.node[self.concat_index].src
                                self.node[index].wr_src_offset = self.node[self.concat_index].concat_offset[order]
                                self.node[index].GlbBufBank.set_wr_mux(self.node[index].wr_start_bank, self.node[self.concat_index].banks_num) 
                                self.GraphBufBank.set_concat(self.node[self.concat_index].src, self.node[self.concat_index].banks_num)
                                self.GraphBufBank.clear_read()
                            
                        else:
                            self.node[index].wr_use_num = self.node[index].upsample_use_num # for upsample
                            self.node[index].wr_start_bank = self.GraphBufBank.find_empty_part(self.node[index].wr_use_num)
                            self.node[index].GlbBufBank.set_wr_mux(self.node[index].wr_start_bank, self.node[index].wr_use_num)
                            self.GraphBufBank.set_new_read(self.node[index].wr_start_bank, self.node[index].wr_use_num)
                            if index > self.concat_index:
                                self.GraphBufBank.clear_concat()
                                
                    else:
                        self.node[index].wr_use_num = self.node[index].upsample_use_num # for upsample
                        self.node[index].wr_start_bank = self.GraphBufBank.find_empty_part(self.node[index].wr_use_num)
                        self.node[index].GlbBufBank.set_wr_mux(self.node[index].wr_start_bank, self.node[index].wr_use_num)
                        self.GraphBufBank.set_new_read(self.node[index].wr_start_bank, self.node[index].wr_use_num)
            
                          
                # set cim_base
                conv_index = self.conv_node.index(index)
                if conv_index == 0:
                    self.node[index].cim_base = self.cim_base
                
                else:
                    self.node[index].cim_base = self.node[self.conv_node[conv_index - 1]].cim_base + self.node[self.conv_node[conv_index - 1]].set_times * self.node[self.conv_node[conv_index - 1]].cim_stride


    def generate_to_code(self, result_path, bias_scale_path):
        fastout_true = "\tol.cpu.fastout_true()\n"
        fastout_false= "\tol.cpu.fastout_false()\n"
        clock_enable = "\tol.npu.set_clock_enable(cim_wr=0, main=1)\n"
        mode = "\tol.cui.set_mode(tx_ddr = 0, tx_enable = 0, rx_ddr = 0, rx_enable =1)\n"
        fence = "\tol.cpu.fence(npu=0x8)\n"
        calc_start = "\tol.npu.set_calc_start()\n"
        
        code_path = result_path + "/python_code.py"
        
        with open(code_path, 'w') as outfile:
            # the prefix part of the python file
            outfile.write("from cim.knight.knight import *\n")
            for i in range(len(self.conv_node)):
                outfile.write("biasLine{0} = []\nscaleLine{0} = []\n".format(i))
                scale_path = bias_scale_path + "/layer{0}_gam.txt".format(i)
                bias_path = bias_scale_path + "/layer{0}_bias.txt".format(i)
                
                outfile.write("with open(r" + "'" + bias_path + "')as fi:\n")
                outfile.write("\tbiasLine{0} = fi.readlines()\n".format(i))
                outfile.write("\tfor i in range(len(biasLine{0})):\n\t\tbiasLine{0}[i] = int(biasLine{0}[i],2)\nfi.close()\n".format(i))
                
                outfile.write("with open(r" + "'" + scale_path + "')as fi:\n")
                outfile.write("\tscaleLine{0} = fi.readlines()\n".format(i))
                outfile.write("\tfor i in range(len(scaleLine{0})):\n\t\tscaleLine{0}[i] = int(scaleLine{0}[i],2)\nfi.close()\n".format(i))
                
            
            outfile.write("def knight_workload(ol):\n")
            #compute part
            flag = 0
            for node in self.node:
                if node.type == "conv":
                    #print(self.node.index(node))
                    outfile.write(fastout_false)
                    outfile.write(clock_enable)
                    outfile.write(mode)
                    node.set_GlbBufBank_param()
                    node.set_param()
                    for GlbBufBank_string in node.set_GlbBufBank_string:
                        outfile.write(GlbBufBank_string)
                    for i in range(len(node.set_param_string)):
                        if flag == 0 :
                            outfile.write(node.set_param_string[i])
                            outfile.write(calc_start)
                            flag = 1
                        else:
                            outfile.write(node.set_param_string[i])
                            outfile.write(fence)
                            outfile.write(calc_start)
                    for i in range(len(node.set_upsample_param_string)):
                        outfile.write(node.set_upsample_param_string[i])
                        outfile.write(fence)
                        outfile.write(calc_start)
                    
                    outfile.write(clock_enable)
                    outfile.write(fence)
                    outfile.write(fastout_true)
                    
            # main function
            outfile.write("def main():\n")
            outfile.write("\tols = KnightChiplet()\n\tknight_workload(ols)\n")
            outfile.write("\tols.dump_inst('test_inst.txt', dir_path=" + "'" + result_path + "'" + ", base=16)\n")
            outfile.write("if __name__ == \"__main__\":\n\tmain()")
            
        outfile.close()

