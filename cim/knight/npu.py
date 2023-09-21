from cim.knight.ip import KnightIP


class KnightNpu(KnightIP):
    def __init__(self, ctrl, base):
        reg = dict(
            status=0x3000,
            cfgSrcImgAddr=0x3001,
            cfgDstImgAddr=0x3002,
            cfgMisc=0x3003,
            glbBankMux=0x3004,
            clockEnable=0x300F,
            cimWrAddr=0x3EFF,
            glbWrData0=0x3F00,
            glbWrData1=0x3F01,
            glbWrData2=0x3F02,
            glbWrData3=0x3F03,
            glbRdData0=0x3F10,
            glbRdData1=0x3F11,
            glbRdData2=0x3F12,
            glbRdData3=0x3F13
        )
        for i in range(32):
            reg['cfgPostBiasScale%d' % i] = 0x3010 + i
        for i in range(15):
            reg['cimWrData%d' % i] = 0x3E10 + i
        super().__init__(ctrl, base, reg)
        self.base_GlbActBuf = self.base + 0x8000

    def set_clock_enable(self, cim_wr, main):
        self.set_reg("clockEnable", (int(cim_wr) << 1) + int(main))

    def get_status(self):
        return self.get_reg('status')

    def set_config_PostScaleBias(self, scale, bias):
        assert isinstance(scale, list) and len(scale) == 32, 'Invalid scale argument'
        assert isinstance(bias, list) and len(bias) == 32, 'Invalid bias argument'
        for i in range(32):
            self.set_config('cfgPostBiasScale%d' % i, (bias[i] << 16) + scale[i])

    def set_config_GlbBufBankMux(self, wr_mux, rd_mux):
        self.set_reg('glbBankMux', (rd_mux << 16) + wr_mux)

    def set_config_SrcImgAddr(self, base, width, height):
        self.set_config('cfgSrcImgAddr', (base << 18) + (width << 9) + height)

    def set_config_DstImgAddr(self, base, width, height):
        self.set_config('cfgDstImgAddr', (base << 18) + (width << 9) + height)

    def set_config_Misc(self, cim_base, post_shift, stride, kernel_size, src_channel_16x, upsample=0, padding=0b0, relu=0,
                        max_pool=0):
        data = relu << 31
        data += max_pool << 30
        data += upsample << 23
        data += cim_base << 16
        data += post_shift << 12
        data += stride << 10
        data += kernel_size << 8
        data += src_channel_16x << 4
        data += padding
        self.set_config('cfgMisc', data)

    def write_GlbActBuf(self, addr, data):
        self.set_reg('glbWrData3', (data >> 96) & 0xFFFF_FFFF)
        self.set_reg('glbWrData2', (data >> 64) & 0xFFFF_FFFF)
        self.set_reg('glbWrData1', (data >> 32) & 0xFFFF_FFFF)
        self.write(self.base_GlbActBuf + addr, data & 0xFFFF_FFFF)

    def set_calc_param(self, src_base, src_width, src_height, dst_base, dst_width, dst_height, src_channel_16x,
                       cim_base, kernel_size, stride, post_shift, post_scale=None, post_bias=None, upsample=0, padding=0x0, relu=0,
                       max_pool=0):
        assert kernel_size == 1 or kernel_size == 3, 'Argument kernel_size can only be 1 or 3, but got %d' % kernel_size
        assert stride == 1 or stride == 2, 'Argument stride can only be 1 or 2, but got %d' % stride
        assert relu == 1 or relu == 0, 'Argument relu can only be 0 or 1, but got %d' % relu
        assert max_pool == 1 or max_pool == 0, 'Argument max_pool can only be 0 or 1, but got %d' % max_pool
        if post_scale is not None and post_bias is not None:
            self.set_config_PostScaleBias(post_scale, post_bias)
        self.set_config_DstImgAddr(dst_base, dst_width, dst_height)
        self.set_config_SrcImgAddr(src_base, src_width, src_height)
        self.set_config_Misc(cim_base, post_shift, stride, kernel_size, src_channel_16x, upsample, padding, relu, max_pool)

    def set_calc_start(self):
        self.set_reg("status", 1 << 31)
