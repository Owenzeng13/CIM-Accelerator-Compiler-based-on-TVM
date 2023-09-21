from cim.knight.ip import KnightIP


class KnightCui(KnightIP):
    def __init__(self, ctrl, base):
        reg = dict(
            cfgEnable=0x0,
            cfgSrcAddr=0x1,
            cfgDstAddr=0x2,
            status=0xF,
            kickoff=0xF,
        )
        super().__init__(ctrl, base, reg)

    def get_status(self):
        return self.get_reg('status')

    def set_mode(self, tx_ddr, tx_enable, rx_ddr, rx_enable):
        data = (int(rx_ddr) << 1) + int(rx_enable)
        data += (int(tx_ddr) << 3) + (int(tx_enable) << 2)
        self.set_config('cfgEnable', data)

    def set_tx_param(self, dst_addr, src_addr, src_len, src_offset=0, src_iter=0):
        self.set_config('cfgSrcAddr', (src_len << 14) + src_addr)
        self.set_config('cfgDstAddr', (dst_addr << 18) + (src_iter << 9) + src_offset)

    def set_tx_start(self):
        self.set_reg('kickoff', 1 << 31)
