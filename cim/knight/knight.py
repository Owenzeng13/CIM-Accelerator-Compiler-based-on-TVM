from cim.knight.cpu import KnightCpu
from cim.knight.npu import KnightNpu
from cim.knight.cui import KnightCui
from pathlib import Path


class KnightChiplet:
    def __init__(self):
        self.cpu = KnightCpu()
        self.npu = KnightNpu(self.cpu, base=0x00000)
        self.cui = KnightCui(self.cpu, base=0x10000)

    def dump_inst(self, file_path, dir_path='./build/', base=2):
        assert base == 2 or base == 16, 'Argument base can only be 2 or 16, but got %d' % base

        dp = Path(dir_path)
        dp.mkdir(parents=True, exist_ok=True)
        p = dp / file_path
        self.cpu.align_inst_len()
        inst = self.cpu.inst

        with open(p, 'w') as fp:
            for j in inst:
                if base == 16:
                    fp.write('%08x\n' % int(j, 2))
                else:
                    fp.write(j + '\n')
