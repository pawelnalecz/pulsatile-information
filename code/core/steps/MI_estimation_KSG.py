# learning.py
from core.step_manager import AbstractStep
from cce import WeightedKraskovEstimator


class Step(AbstractStep):

    step_name = 'MIksg'

    required_parameters = ['nearest_neighbor_k']
    input_files = ['extracted_slices']
    output_files = {'mutual_information': '.pkl.gz', 'mutual_informations': '.pkl.gz'}#, 'prediction_probas': '.pkl.gz'}



    def perform(self, **kwargs):
        print('------ESTIMATING MI (KSG)------')

        tpos = kwargs['target_position']
        nearest_neighbor_k = kwargs['nearest_neighbor_k']

        slices = self.load_file('extracted_slices')

        print('Reformating data', end='... ', flush=True)
        data = list(zip(1 * (slices['target']), slices['flat_data']))#[(1 * (row['target'] == tpos), np.array(row['flat_data'])) for idx,row in slices.iterrows()]

        print('Computing MI', end='... ', flush=True)
        mutual_information = WeightedKraskovEstimator(data).calculate_mi(k=nearest_neighbor_k)

        print('done.')

        self.save_file(mutual_information, 'mutual_information')
        self.save_file([mutual_information], 'mutual_informations')

