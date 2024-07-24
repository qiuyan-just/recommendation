from problem import Problem
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from config import Config
from dispatch import Dispatch
import pandas as pd


class Solver(object):
    def __init__(self, problem: Problem):
        self.problem = problem
        self.alpha = {}
        self.x_ijt_var = None
        self.dispatch_dict = {}
        self.model = gp.Model()
        self.beta = {}
        self.total_obj = 0

    def solve_primal_dual_model(self,lr):
        total_obj = 0
        for t1, patients in self.problem.batch_patient.items():
            for patient in patients:
                T = list(range(t1, t1+Config.period))
                self.model = gp.Model()
                self.x_ijt_var = self.model.addVars(self.problem.doctor_id_list, T, vtype=GRB.BINARY)
                obj_sum = []
                for doctor_id in self.problem.doctor_id_list:
                    for t in T:
                        obj_sum += self.problem.matching_dict[patient.id][doctor_id] * self.x_ijt_var[doctor_id, t]
                self.model.setObjective(obj_sum, GRB.MAXIMIZE)

                """set constraints"""
                self.model.addConstrs((self.x_ijt_var[doctor_id, t] <= self.problem.doctor_dict[doctor_id].capacity_t[t]
                                      for doctor_id in self.problem.doctor_id_list for t in T), name="capacity_constr")
                self.model.addConstr((gp.quicksum(self.x_ijt_var[doctor_id, t] for doctor_id in self.problem.doctor_id_list
                                      for t in T) <= 1), name="assign_constr")

                """solve"""
                self.model.update()
                self.model.optimize()
                assign_doctor_id = None
                assign_t = None
                for t in T:
                    for doctor_id in self.problem.doctor_id_list:
                        if self.x_ijt_var[doctor_id, t].x > .9:
                            assign_t = t
                            assign_doctor_id = doctor_id
                if assign_doctor_id:
                    reduce_cost = -1
                    if assign_doctor_id in self.alpha and assign_t in self.alpha[assign_doctor_id]:
                        reduce_cost = self.problem.matching_dict[patient.id][assign_doctor_id] - self.alpha[assign_doctor_id][assign_t]
                    else:
                        reduce_cost = self.problem.matching_dict[patient.id][assign_doctor_id]
                    if reduce_cost >= 0:
                        self.beta[patient.id] = reduce_cost
                        total_obj += self.problem.matching_dict[patient.id][assign_doctor_id]
                        dispatch_model = Dispatch(patient.id, assign_doctor_id, assign_t, self.problem.t_date_dict[assign_t])
                        doctor = self.problem.doctor_dict[assign_doctor_id]
                        if assign_doctor_id not in self.alpha:
                            self.alpha[assign_doctor_id] = {}
                        if assign_t not in self.alpha[assign_doctor_id]:
                            self.alpha[assign_doctor_id][assign_t] = 0
                        self.alpha[assign_doctor_id][assign_t] = self.alpha[assign_doctor_id][assign_t] * (
                                    1 + 1 / self.problem.doctor_dict[assign_doctor_id].capacity_t[assign_t]) \
                                                                 + lr * (self.problem.matching_dict[patient.id][assign_doctor_id] /
                                                                             self.problem.doctor_dict[assign_doctor_id].capacity_t[assign_t])
                        # self.alpha[assign_doctor_id][assign_t] = self.alpha[assign_doctor_id][assign_t] * (1 + 1 / self.problem.doctor_dict[assign_doctor_id].capacity_t[assign_t]) \
                        #     + Config.gamma * (self.problem.matching_dict[patient.id][assign_doctor_id] / self.problem.doctor_dict[assign_doctor_id].capacity_t[assign_t])
                        if doctor.capacity_t[assign_t] > 0:
                            doctor.update_capacity_t(assign_t, doctor.capacity_t[assign_t]-1)
                        self.dispatch_dict[patient.id] = dispatch_model
                    else:
                        self.beta[patient.id] = 0
                        if assign_doctor_id not in self.alpha:
                            self.alpha[assign_doctor_id] = {}
                        if assign_t not in self.alpha[assign_doctor_id]:
                            self.alpha[assign_doctor_id][assign_t] = 0
                else:
                    self.beta[patient.id] = 0
        writer = pd.ExcelWriter('./output/results/primal_dual_output.xlsx')
        patients_df = pd.read_csv(Config.patient_file_path)
        doctor_id_list = []
        service_time_list = []
        beta_list = []
        for index, row in patients_df.iterrows():
            patient_id = row["id"]
            if patient_id in self.beta:
                beta_list.append(self.beta[patient_id])
            else:
                beta_list.append(0)
            if patient_id in self.dispatch_dict:
                dispatch_model = self.dispatch_dict[patient_id]
                doctor_id_list.append(dispatch_model.doctor_id)
                service_time_list.append(dispatch_model.data)
            else:
                doctor_id_list.append(np.nan)
                service_time_list.append(np.nan)
        patients_df["doctor_id"] = doctor_id_list
        patients_df["service_time"] = service_time_list
        patients_df.to_excel(writer, sheet_name="assign_solution", index=False)
        # patients_df.to_csv("primal_dual_output.csv", index=False)
        df_beta = pd.DataFrame()
        for patient_id, beta in self.beta.items():
            df_beta = df_beta.append({"patient_id": patient_id, "beta": beta}, ignore_index=True)
        df_beta = df_beta[["patient_id", "beta"]]
        df_beta.to_excel(writer, sheet_name="beta", index=False)
        df_alpha = pd.DataFrame()
        for doctor_id, t_set in self.alpha.items():
            for t, alpha in t_set.items():
                df_alpha = df_alpha.append({"doctor_id": doctor_id, "t": t, "data": self.problem.t_date_dict[t], "alpha": alpha}, ignore_index=True)
        df_alpha = df_alpha[["doctor_id", "t", "data", "alpha"]]
        df_alpha.to_excel(writer, sheet_name="alpha", index=False)
        print("primal dual模型最终的目标函数值为：", total_obj)
        writer.save()

    def solve_primal_dual_t1(self, patients, t_1, lr):
        unassigned_patients = []
        for patient in patients:
            T = list(range(t_1, min(t_1 + Config.period, Config.total_period + 1)))
            self.model = gp.Model()
            self.x_ijt_var = self.model.addVars(self.problem.doctor_id_list, T, vtype=GRB.BINARY)
            obj_sum = []
            for doctor_id in self.problem.doctor_id_list:
                for t in T:
                    obj_sum += self.problem.matching_dict[patient.id][doctor_id] * self.x_ijt_var[doctor_id, t]
            if not obj_sum:
                print("obj_sum is: ", obj_sum)
                for doctor_id in self.problem.doctor_id_list:
                    for t in T:
                        print(self.problem.matching_dict[patient.id][doctor_id])
            self.model.setObjective(obj_sum, GRB.MAXIMIZE)

            """set constraints"""
            self.model.addConstrs((self.x_ijt_var[doctor_id, t] <= self.problem.doctor_dict[doctor_id].capacity_t[t]
                                   for doctor_id in self.problem.doctor_id_list for t in T), name="capacity_constr")
            self.model.addConstr(
                (gp.quicksum(self.x_ijt_var[doctor_id, t] for doctor_id in self.problem.doctor_id_list
                             for t in T) <= 1), name="assign_constr")

            """solve"""
            self.model.update()
            self.model.optimize()
            assign_doctor_id = None
            assign_t = None
            for t in T:
                for doctor_id in self.problem.doctor_id_list:
                    if self.x_ijt_var[doctor_id, t].x > .9:
                        assign_t = t
                        assign_doctor_id = doctor_id
            if assign_doctor_id:
                reduce_cost = -1
                if assign_doctor_id in self.alpha and assign_t in self.alpha[assign_doctor_id]:
                    reduce_cost = self.problem.matching_dict[patient.id][assign_doctor_id] - \
                                  self.alpha[assign_doctor_id][assign_t]
                else:
                    reduce_cost = self.problem.matching_dict[patient.id][assign_doctor_id]
                if reduce_cost >= 0:
                    self.beta[patient.id] = reduce_cost
                    self.total_obj += self.problem.matching_dict[patient.id][assign_doctor_id]
                    dispatch_model = Dispatch(patient.id, assign_doctor_id, assign_t,
                                              self.problem.t_date_dict[assign_t])
                    doctor = self.problem.doctor_dict[assign_doctor_id]
                    if assign_doctor_id not in self.alpha:
                        self.alpha[assign_doctor_id] = {}
                    if assign_t not in self.alpha[assign_doctor_id]:
                        self.alpha[assign_doctor_id][assign_t] = 0

                    self.alpha[assign_doctor_id][assign_t] = self.alpha[assign_doctor_id][assign_t] * (
                            1 + 1 / self.problem.doctor_dict[assign_doctor_id].capacity_t[assign_t]) \
                                                             + lr * (self.problem.matching_dict[patient.id][
                                                                         assign_doctor_id] /self.problem.doctor_dict[
                                                                         assign_doctor_id].capacity_t[assign_t])
                    # self.alpha[assign_doctor_id][assign_t] = self.alpha[assign_doctor_id][assign_t] * (
                    #         1 + 1 / self.problem.doctor_dict[assign_doctor_id].capacity_t[assign_t]) \
                    #                                          + Config.gamma * (
                    #                                                  self.problem.matching_dict[patient.id][
                    #                                                      assign_doctor_id] /
                    #                                                  self.problem.doctor_dict[
                    #                                                      assign_doctor_id].capacity_t[assign_t])
                    if doctor.capacity_t[assign_t] > 0:
                        doctor.update_capacity_t(assign_t, doctor.capacity_t[assign_t] - 1)
                    self.dispatch_dict[patient.id] = dispatch_model
                else:
                    self.beta[patient.id] = 0
                    if assign_doctor_id not in self.alpha:
                        self.alpha[assign_doctor_id] = {}
                    if assign_t not in self.alpha[assign_doctor_id]:
                        self.alpha[assign_doctor_id][assign_t] = 0
            else:
                self.beta[patient.id] = 0
                unassigned_patients.append(patient)
        return unassigned_patients

    def solve_primal_dual_model_v2(self,lr):
        iteration = 1
        for t1, patients in self.problem.batch_patient.items():
            t_1 = t1
            unassigned_patients = self.solve_primal_dual_t1(patients, t_1,lr)
            while iteration <= Config.iteration and len(unassigned_patients) > 1:
                iteration += 1
                t_1 += 1
                unassigned_patients = self.solve_primal_dual_t1(unassigned_patients, t_1,lr)

        writer = pd.ExcelWriter('./output/results/primal_dual_v2_output.xlsx')
        patients_df = pd.read_csv(Config.patient_file_path)
        doctor_id_list = []
        service_time_list = []
        beta_list = []
        for index, row in patients_df.iterrows():
            patient_id = row["id"]
            if patient_id in self.beta:
                beta_list.append(self.beta[patient_id])
            else:
                beta_list.append(0)
            if patient_id in self.dispatch_dict:
                dispatch_model = self.dispatch_dict[patient_id]
                doctor_id_list.append(dispatch_model.doctor_id)
                service_time_list.append(dispatch_model.data)
            else:
                doctor_id_list.append(np.nan)
                service_time_list.append(np.nan)
        patients_df["doctor_id"] = doctor_id_list
        patients_df["service_time"] = service_time_list
        patients_df.to_excel(writer, sheet_name="assign_solution", index=False)
        df_beta = pd.DataFrame()
        for patient_id, beta in self.beta.items():
            df_beta = df_beta.append({"patient_id": patient_id, "beta": beta}, ignore_index=True)
        df_beta = df_beta[["patient_id", "beta"]]
        df_beta.to_excel(writer, sheet_name="beta", index=False)
        df_alpha = pd.DataFrame()
        for doctor_id, t_set in self.alpha.items():
            for t, alpha in t_set.items():
                df_alpha = df_alpha.append(
                    {"doctor_id": doctor_id, "t": t, "data": self.problem.t_date_dict[t], "alpha": alpha},
                    ignore_index=True)
        df_alpha = df_alpha[["doctor_id", "t", "data", "alpha"]]
        df_alpha.to_excel(writer, sheet_name="alpha", index=False)
        print("primal dual v2 模型最终的目标函数值为：", self.total_obj)
        writer.save()


    def solve_total_model(self):
        self.model = gp.Model()
        self.x_ijt_var = self.model.addVars(self.problem.patient_id_list, self.problem.doctor_id_list, self.problem.t_list,
                                            vtype=GRB.BINARY)
        obj_sum = []
        for t in self.problem.t_list:
            for patient_id in self.problem.patient_id_list:
                for doctor_id in self.problem.doctor_id_list:
                    matching_r = self.problem.matching_dict[patient_id][doctor_id]
                    obj_sum += matching_r * self.x_ijt_var[patient_id, doctor_id, t]
        """set obj"""
        self.model.setObjective(obj_sum, GRB.MAXIMIZE)
        """set constraint"""
        self.model.addConstrs((gp.quicksum(self.x_ijt_var[patient_id, doctor_id, t] for patient_id in self.problem.patient_id_list) <=
                               self.problem.doctor_dict[doctor_id].capacity_t[t] for doctor_id in self.problem.doctor_id_list for t in self.problem.t_list),
                              name="capacity_constr")
        self.model.addConstrs((gp.quicksum(self.x_ijt_var[patient_id, doctor_id, t] for doctor_id in self.problem.doctor_id_list for t in self.problem.t_list) <= 1
                               for patient_id in self.problem.patient_id_list),
                              name="assign_constr")
        for t in self.problem.t_list:
            for patient_id in self.problem.patient_id_list:
                for doctor_id in self.problem.doctor_id_list:
                    if t < self.problem.patient_dict[patient_id].arrival_time_t:
                        self.model.addConstr((self.x_ijt_var[patient_id, doctor_id, t] == 0), name="left_time_window_constr")
                    else:
                        self.model.addConstr((t * self.x_ijt_var[patient_id, doctor_id, t] >= self.problem.patient_dict[patient_id].arrival_time_t * self.x_ijt_var[patient_id, doctor_id, t]),
                                             name="left_time_window_constr")
        # self.model.addConstrs(
        #     (t * self.x_ijt_var[patient_id, doctor_id, t] <= self.problem.patient_dict[patient_id].arrival_time_t + Config.total_period
        #      for t in self.problem.t_list for patient_id in self.problem.patient_id_list for doctor_id in
        #      self.problem.doctor_id_list),
        #     name="right_time_window_constr")

        ## 表示与滚动时间的方法比较，时间也应该增加对应的iteration
        # self.model.addConstrs(
        #     (t * self.x_ijt_var[patient_id, doctor_id, t] <=  Config.total_period
        #      for t in self.problem.t_list for patient_id in self.problem.patient_id_list for doctor_id in
        #      self.problem.doctor_id_list),
        #     name="right_time_window_constr")

        ## 表示一般的决策周期period
        self.model.addConstrs(
            (t * self.x_ijt_var[patient_id, doctor_id, t] <= Config.period
             for t in self.problem.t_list for patient_id in self.problem.patient_id_list for doctor_id in
             self.problem.doctor_id_list),
            name="right_time_window_constr")

        self.model.update()
        self.model.optimize()
        self.model.write("model.lp")
        if self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")
        patients_df = pd.read_csv(Config.patient_file_path)
        doctor_id_list = []
        service_time_list = []
        for index, row in patients_df.iterrows():
            patient_id = row["id"]
            find = False
            for doctor_id in self.problem.doctor_id_list:
                for t in self.problem.t_list:
                    if self.x_ijt_var[patient_id, doctor_id, t].x > .9:
                        doctor_id_list.append(doctor_id)
                        service_time_list.append(self.problem.t_date_dict[t])
                        find = True
                        break
                if find:
                    break
            if not find:
                doctor_id_list.append(np.nan)
                service_time_list.append(np.nan)
        patients_df["doctor_id"] = doctor_id_list
        patients_df["service_time"] = service_time_list
        patients_df.to_excel("./output/results/total_model_output.xlsx", index=False)

    def greedy_solver(self):
        total_obj = 0
        for t1, patients in self.problem.batch_patient.items():
            for patient in patients:
                T = list(range(t1, t1 + Config.period))
                assign_doctor_id = None
                assign_t = None
                find_matching_doctor = False
                for doctor_matching in self.problem.patient_doctor_matching_r[patient.id]:
                    doctor = self.problem.doctor_dict[doctor_matching[0]]
                    for t in T:
                        if doctor.capacity_t[t] > 0:
                            assign_doctor_id = doctor.id
                            assign_t = t
                            find_matching_doctor = True
                            break
                    if find_matching_doctor:
                        break
                if assign_doctor_id:
                    total_obj += self.problem.matching_dict[patient.id][assign_doctor_id]
                    dispatch_model = Dispatch(patient.id, assign_doctor_id, assign_t,
                                              self.problem.t_date_dict[assign_t])
                    doctor = self.problem.doctor_dict[assign_doctor_id]
                    if doctor.capacity_t[assign_t] > 0:
                        doctor.update_capacity_t(assign_t, doctor.capacity_t[assign_t] - 1)
                    self.dispatch_dict[patient.id] = dispatch_model

        writer = pd.ExcelWriter('./output/results/greedy_output.xlsx')
        patients_df = pd.read_csv(Config.patient_file_path)
        doctor_id_list = []
        service_time_list = []
        for index, row in patients_df.iterrows():
            patient_id = row["id"]
            if patient_id in self.dispatch_dict:
                dispatch_model = self.dispatch_dict[patient_id]
                doctor_id_list.append(dispatch_model.doctor_id)
                service_time_list.append(dispatch_model.data)
            else:
                doctor_id_list.append(np.nan)
                service_time_list.append(np.nan)
        patients_df["doctor_id"] = doctor_id_list
        patients_df["service_time"] = service_time_list
        patients_df.to_excel(writer, sheet_name="assign_solution", index=False)
        print("greedy方案最终的目标函数值为：", total_obj)
        writer.save()

    def threshold_solver(self):
        total_obj = 0
        self.problem.shuffle_patient_doctor_matching_r()
        for t1, patients in self.problem.batch_patient.items():
            for patient in patients:
                T = list(range(t1, t1 + Config.period))
                assign_doctor_id = None
                assign_t = None
                find_matching_doctor = False
                for doctor_matching in self.problem.patient_doctor_matching_r[patient.id]:
                    doctor = self.problem.doctor_dict[doctor_matching[0]]
                    for t in T:
                        if doctor.capacity_t[t] > 0 and doctor_matching[1] >= Config.threshold:
                            assign_doctor_id = doctor.id
                            assign_t = t
                            find_matching_doctor = True
                            break
                    if find_matching_doctor:
                        break
                if assign_doctor_id:
                    total_obj += self.problem.matching_dict[patient.id][assign_doctor_id]
                    dispatch_model = Dispatch(patient.id, assign_doctor_id, assign_t,
                                              self.problem.t_date_dict[assign_t])
                    doctor = self.problem.doctor_dict[assign_doctor_id]
                    if doctor.capacity_t[assign_t] > 0:
                        doctor.update_capacity_t(assign_t, doctor.capacity_t[assign_t] - 1)
                    self.dispatch_dict[patient.id] = dispatch_model

        writer = pd.ExcelWriter('./output/results/threshold_output.xlsx')
        patients_df = pd.read_csv(Config.patient_file_path)
        doctor_id_list = []
        service_time_list = []
        for index, row in patients_df.iterrows():
            patient_id = row["id"]
            if patient_id in self.dispatch_dict:
                dispatch_model = self.dispatch_dict[patient_id]
                doctor_id_list.append(dispatch_model.doctor_id)
                service_time_list.append(dispatch_model.data)
            else:
                doctor_id_list.append(np.nan)
                service_time_list.append(np.nan)
        patients_df["doctor_id"] = doctor_id_list
        patients_df["service_time"] = service_time_list
        patients_df.to_excel(writer, sheet_name="assign_solution", index=False)
        print("threshold方案最终的目标函数值为：", total_obj)
        writer.save()
