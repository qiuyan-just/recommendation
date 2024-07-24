from datetime import datetime


class Patient(object):
    def __init__(self, p_id: str, arrival_time: datetime, arrival_time_t: int):
        self.id = p_id
        self.arrival_time = arrival_time
        self.arrival_time_t = arrival_time_t
