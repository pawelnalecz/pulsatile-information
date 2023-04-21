import numpy as np

from core.step_manager import AbstractStep
from utils.math_utils import plogp
import pandas as pd




class Step(AbstractStep):

    step_name = 'MIcm'

    required_parameters = []
    input_files = ['confusion_matrix']
    output_files = {'mutual_information': '.pkl.gz', 'mutual_informations': '.pkl.gz'}



    def perform(self, **kwargs):
        print('------ESTIMATING MI (CONFUSION MATRIX)------')

        confusion_matrix: pd.DataFrame = self.load_file('confusion_matrix')

        mutual_informations = np.array([
            print(conf_m) or (plogp(conf_m.groupby('y_true').sum()).sum() + plogp(conf_m.groupby('y_pred').sum()).sum() - plogp(conf_m.groupby(['y_pred', 'y_true']).sum()).sum() - plogp(conf_m.sum())) / conf_m.sum()
            for _, conf_m in confusion_matrix.groupby('iteration')
            ])

        mutual_information = np.mean(mutual_informations)

        print(mutual_informations)
        self.save_file(mutual_information, 'mutual_information')
        self.save_file(mutual_informations, 'mutual_informations')




