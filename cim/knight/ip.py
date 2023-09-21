class KnightIP:
    def __init__(self, ctrl, base, reg=None):
        self.c = ctrl
        self.base = base
        self.reg = dict() if reg is None else dict(reg)

        self.cfgCache = dict()
        for k in self.reg.keys():
            if k.startswith("cfg"):
                self.cfgCache.setdefault(k, None)

    def write(self, offset, data):
        addr = offset + self.base
        self.c.write_bus(addr, data)

    def set_reg(self, key, data):
        self.write(self.reg[key], data)

    def get_reg(self, key):
        addr = self.reg[key] + self.base
        return self.c.read_bus(addr)

    def set_config(self, key, data):
        if data != self.cfgCache[key]:
            self.cfgCache[key] = data
            self.set_reg(key, data)
