"""
ConnectionCP.py

This code connect from Python3 to Coppeliasim

Author:
    Mario Pastrana (mariopastrana403@gmail.com)
    EVA PROJECT - Universidade de Brasília

Version:
    0.0.1 (beta)

Release Date:
    Jan 23, 2023
"""


import matplotlib.pyplot as plt
import numpy as np
import math as mat
import Bioinspired
import csv

class EVA_robot():

    def __init__(self):

        self.Ki = 0
        self.Kd = 0
        self.Kp = 0
        self.Sample_Time = 0.05
        self.unomenosalfa = 0
        self.alfa = 0
        self.Min_value = 0
        self.Conv_array = []

    def Trainig_PID(self, selAlg):

        ctf = Bioinspired.Bioinspired()

        if 'PSO_' in selAlg:
            ##### Parametros PSO ####
            S = 7
            N = 3
            maxIter = 20
            w0 = 0.9
            wf = 0.1
            c1 = 2.05
            c2 = 2.05
            vMax = 3
            vIni = vMax / 10
            xMax = 3
            xMin = 0.001
            ##### Parametros Funciones de transferencia ####
            Sample_time = 0.05
            ctf.PSO(S, N, maxIter, Sample_time, w0, wf, c1, c2, vMax, vIni, xMax, xMin)

        elif 'ARPSO' in selAlg:

            ##### Parametros ARPSO ####
            S = 7
            N = 3
            maxIter = 20
            w0 = 0.9
            wf = 0.1
            c1 = 2.05
            c2 = 2.05
            vMax = 3
            vIni = vMax / 10
            xMax = 2
            xMin = 0.001
            para_div = 4
            para_iter = 50  # em porcentagem
            Sample_time = 0.05
            ctf.ARPSO(S, N, maxIter, Sample_time, w0, wf, c1, c2, vMax, vIni, xMax, xMin, para_div, para_iter)

        elif 'ABC' in selAlg:
            ##### Parametros ABC ####
            S = 7
            N = 3
            Pm = 0.6
            Pl = 0.1
            maxIter = 20
            xMax = 2
            xMin = 0.001
            Sample_time = 0.05
            ctf.ABC(S, N, maxIter, Sample_time, xMax, xMin, Pm, Pl)

        elif 'DE' in selAlg:
            ##### Parametros ABC ####
            S = 7
            N = 3
            maxIter = 20
            xMax = 2
            xMin = 0.001
            Sample_time = 0.05
            F_mutation = 1.25
            crossover = 0.75
            ctf.DE(S, maxIter, N, F_mutation, crossover, xMin, xMax, Sample_time)

        elif 'MFO' in selAlg:
            ##### Parametros ABC ####
            S = 7
            N = 3
            maxIter = 20
            xMax = 2
            xMin = 0.001
            Sample_time = 0.05
            ctf.MFO(S, maxIter, N, Sample_time, xMin, xMax)

        elif 'GWO' in selAlg:
            ##### Parametros ABC ####
            S = 7
            N = 3
            maxIter = 20
            xMax = 2
            xMin = 0.001
            Sample_time = 0.05
            ctf.GWO(S, maxIter, N, Sample_time, xMin, xMax)

        elif 'WOA' in selAlg:
            ##### Parametros ABC ####
            S = 7
            N = 3
            maxIter = 20
            xMax = 2
            xMin = 0.001
            Sample_time = 0.05
            ctf.WOA(S, maxIter, N, Sample_time, xMin, xMax)

        elif 'BAT' in selAlg:
            ##### Parametros ABC ####
            S = 7
            N = 3
            maxIter = 20
            xMax = 2
            xMin = 0.001
            Sample_time = 0.05
            ctf.BAT(S, maxIter, N, Sample_time, xMin, xMax)

        elif 'AHA' in selAlg:
            ##### Parametros ABC ####
            S = 7
            N = 3
            maxIter = 20
            xMax = [2]
            xMin = [0.001]
            Sample_time = 0.05
            ctf.AHA(S, maxIter, N, Sample_time, xMin, xMax)



        self.Kp = ctf.ys[0]
        self.Ki = ctf.ys[1]
        self.Kd = ctf.ys[2]
        raizes = np.roots([self.Kd, self.Kp, self.Ki])
        absoluto = abs(raizes)
        mayor = max(absoluto)
        Filter_e = 1 / (mayor * 10)
        self.unomenosalfa = mat.exp(-(self.Sample_Time / Filter_e))
        self.alfa = 1 - self.unomenosalfa
        self.Min_value = ctf.fitVector[-1]
        self.Conv_array = ctf.fitVector


if __name__ == "__main__":

    # alg_implemented = "PSO_"
    # alg_implemented = "DE"
    # alg_implemented = "ABC"
    # alg_implemented = "ARPSO"
    # alg_implemented = "MFO"
    alg_implemented = "GWO"
    # alg_implemented = "WOA"
    # alg_implemented = "BAT"
    # alg_implemented = "AHA"

    with open(f'variables_{alg_implemented}.csv', 'w', newline='') as csvfile:

        number_of_exp = 3
        best_min = 100000
        EVA_object = EVA_robot()
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [f'Kp_{alg_implemented}', f'Ki_{alg_implemented}', f'Kd_{alg_implemented}', '1 - alfa', 'alfa', 'Num_exp', 'Min_Global_exp', 'Best_exp', 'Best_exp_min'])

        for i in range(number_of_exp):

            print(f'Experiment_{i}')
            EVA_object.Trainig_PID(alg_implemented)
            kp_BIO = EVA_object.Kp
            ki_BIO = EVA_object.Ki
            kd_BIO = EVA_object.Kd
            unomenosalfa_BIO = EVA_object.unomenosalfa
            alfa = EVA_object.alfa
            Min_global = EVA_object.Min_value
            if Min_global < best_min:
                best_exp = i
                best_min = Min_global

            csv_writer.writerow([kp_BIO, ki_BIO, kd_BIO, unomenosalfa_BIO, alfa, i, Min_global, best_exp, best_min])



