import numpy as np

from core.step_manager import AbstractStep
from utils.math_utils import plogp
import pandas as pd




class Step(AbstractStep):

    step_name = 'MIcmg'

    required_parameters = ['fields_for_mi_computation_grouping']
    input_files = ['confusion_matrix']
    output_files = {'mutual_information_grouped': '.pkl.gz', 'mutual_informations_grouped': '.pkl.gz'}



    def perform(self, **kwargs):
        print('------ESTIMATING MI (FROM CONFUSION MATRIX, GROUPED)------')

        fields_for_mi_computation_grouping = kwargs['fields_for_mi_computation_grouping']

        confusion_matrix: pd.DataFrame = self.load_file('confusion_matrix')

        def compute_mi(conf_m: pd.DataFrame):
            return (plogp(conf_m.groupby('y_true').sum()).sum() + plogp(conf_m.groupby('y_pred').sum()).sum() - plogp(conf_m.groupby(['y_pred', 'y_true']).sum()).sum() - plogp(conf_m.sum())) / conf_m.sum()

        mutual_informations = confusion_matrix.groupby(fields_for_mi_computation_grouping).agg(compute_mi)
        mutual_information = mutual_informations.mean()

        print(mutual_informations)
        self.save_file(mutual_information, 'mutual_information_grouped')
        self.save_file(mutual_informations, 'mutual_informations_grouped')




