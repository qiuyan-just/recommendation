

class Doctor(object):
    def __init__(self, d_id, capacity):
        self.id = d_id
        self.capacity = capacity
        self.capacity_t = {}

    def update_capacity_t(self, t, capacity):
        self.capacity_t[t] = capacity
