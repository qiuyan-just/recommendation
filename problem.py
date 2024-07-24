import pandas as pd
from config import Config
from datetime import datetime, timedelta
from patient import Patient
from doctor import Doctor
import random


class Problem(object):
    def __init__(self):
        self.t_date_dict = {}
        self.date_t_dict = {}
        self.batch_patient = {}
        self.t_list = []
        self.matching_dict = {}
        self.patient_id_list = []
        self.patient_dict = {}
        self.doctor_id_list = []
        self.doctor_dict = {}
        self.patient_doctor_matching_r = {}

    def build(self):
        patients_df = pd.read_csv(Config.patient_file_path)
        current_t = 1
        for index, row in patients_df.iterrows():
            patient_id = row["id"]
            arrival_time_data = datetime.strptime(row["arrival time"], "%Y/%m/%d")
            arrival_time = row["arrival time"]
            patient_t = current_t
            arrival_time_cur = arrival_time
            if arrival_time not in self.date_t_dict:
                self.t_date_dict[current_t] = arrival_time
                self.date_t_dict[arrival_time] = current_t
                self.t_list.append(current_t)
                for i in range(Config.total_period):
                    arrival_time_data = arrival_time_data + timedelta(days=1)
                    current_t += 1
                    self.t_list.append(current_t)
                    self.t_date_dict[current_t] = arrival_time_data.strftime("%Y/%m/%d")
                    self.date_t_dict[arrival_time_data.strftime("%Y/%m/%d")] = current_t
                current_t += 1
            else:
                for i in range(Config.total_period):
                    arrival_time_data = arrival_time_data + timedelta(days=1)
                    if arrival_time_data.strftime("%Y/%m/%d") in self.date_t_dict:
                        continue
                    self.t_list.append(current_t)
                    self.t_date_dict[current_t] = arrival_time_data.strftime("%Y/%m/%d")
                    self.date_t_dict[arrival_time_data.strftime("%Y/%m/%d")] = current_t
                    current_t += 1
                patient_t = self.date_t_dict[arrival_time]
            patient = Patient(patient_id, arrival_time_cur, patient_t)
            self.patient_id_list.append(patient_id)
            self.patient_dict[patient_id] = patient
            if patient_t not in self.batch_patient:
                self.batch_patient[patient_t] = []
            self.batch_patient[patient_t].append(patient)
        doctor_df = pd.read_csv(Config.doctor_file_path)
        for index, row in doctor_df.iterrows():
            doctor_id = row["id"]
            capacity = row["capacity"]
            doctor = Doctor(doctor_id, capacity)
            self.doctor_id_list.append(doctor_id)
            self.doctor_dict[doctor_id] = doctor
            for t in self.t_date_dict:
                doctor.update_capacity_t(t, capacity)
        matching_df = pd.read_csv(Config.matching_file_path)
        for index, row in matching_df.iterrows():
            patient_id = row["patients"]
            doctor_id = row["doctors"]
            matching_r = row["r_matching"]
            if patient_id not in self.patient_doctor_matching_r:
                self.patient_doctor_matching_r[patient_id] = []
            self.patient_doctor_matching_r[patient_id].append((doctor_id, matching_r))
            if patient_id not in self.matching_dict:
                self.matching_dict[patient_id] = {}
            if doctor_id not in self.matching_dict[patient_id]:
                self.matching_dict[patient_id][doctor_id] = matching_r

        for patient_id, matching_rs in self.patient_doctor_matching_r.items():
            matching_rs.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place 原地排序

    def shuffle_patient_doctor_matching_r(self):
        random.seed(0)
        for patient_id, matching_rs in self.patient_doctor_matching_r.items():
            random.shuffle(matching_rs)
