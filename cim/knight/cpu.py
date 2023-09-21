def int2str(num, pad_zero_to_size):
    num_range = 2 ** pad_zero_to_size
    if num < 0:
        num = num_range + num
    s = "{0:b}".format(num)
    if len(s) > pad_zero_to_size:
        raise OverflowError("Num is too large to convert to a {} number".format(pad_zero_to_size))
    elif len(s) < pad_zero_to_size:
        s = '0' * (pad_zero_to_size - len(s)) + s
    return s


class _KnightCpuBase:
    def __init__(self):
        self.inst = list()

    def append(self, s):
        self.inst.append(s.strip())

    def _IMM2RGH(self, imm):
        s = "0000"  # instType, 4 bits, 31:28
        s += "0000"  # DontCare, 4 bits, 27:24
        s += int2str(imm, 24)  # imm, 24 bits, 23:0
        self.append(s)

    def _IMM2RGL2ADD(self, imm, addr):
        s = "0001"  # instType, 4 bits, 31:28
        s += int2str(addr, 20)  # addr, 20 bits, 27:8
        s += int2str(imm, 8)  # imm, 8 bits, 7:0
        self.append(s)

    def _ADD2REG(self, addr):
        s = "0010"  # instType, 4 bits, 31:28
        s += int2str(addr, 20)  # addr, 20 bits, 27:8
        s += "0" * 8  # DontCare, 8 bits, 7:0
        self.append(s)

    def _FCEMORT(self, npu=None, cui=None, gpio=None):
        s = "0011"  # instType, 4 bits, 31:28
        s += "0"  # DontCare, 1 bit, 27
        s += "0" if gpio else "1"  # GPIO mask, 1 bit, 26
        s += "0" if cui else "1"  # CUI mask, 1 bit, 25
        s += "0" if npu else "1"  # NPU mask, 1 bit, 24
        s += "0" * 12  # DontCare, 12 bits, 23:12
        if gpio:  # GPIO mort, 4 bits, 11:8
            s += int2str(gpio, 4)
        else:
            s += "0000"
        if cui:  # CUI mort, 4 bits, 7:4
            s += int2str(cui, 4)
        else:
            s += "0000"
        if npu:  # NPU mort, 4 bits, 3:0
            s += int2str(npu, 4)
        else:
            s += "0000"
        self.append(s)

    def _FASTOUT(self, bit):
        s = "0100"  # instType, 4 bits, 31:28
        s += "0" * 27  # DontCare, 27 bits, 27:1
        s += int2str(bit, 1)  # Bit, 1 bit, 0
        self.append(s)

    def _REG2ADD(self, addr):
        s = "0101"  # instType, 4 bits, 31:28
        s += int2str(addr, 20)  # addr, 20 bits, 27, 8
        s += "0" * 8
        self.append(s)


class KnightCpu(_KnightCpuBase):
    def write_bus(self, addr, imm=None):
        if imm is None:
            # if imm==None, write Reg to [addr]
            self._REG2ADD(addr)
        else:
            imm_l = imm & 0xFF
            imm_h = imm >> 8
            # if imm_h != 0:
            #     self._IMM2RGH(imm_h)
            self._IMM2RGH(imm_h)
            self._IMM2RGL2ADD(imm_l, addr)

    def read_bus(self, addr):
        self._ADD2REG(addr)

    def fastout_true(self):
        self._FASTOUT(1)

    def fastout_false(self):
        self._FASTOUT(0)

    def fence(self, npu=None, cuiTx=None, cuiRx=None):
        self._FCEMORT(npu, cuiTx, cuiRx)

    def align_inst_len(self, n=16):
        if len(self.inst) % n != 0:
            for _ in range(n - len(self.inst)):
                self.inst.append('0' * 32)

    def curr_inst_len(self):
        return len(self.inst)
