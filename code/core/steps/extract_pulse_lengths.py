import pandas as pd
import numpy as np

from core.step_manager import AbstractStep

class Step(AbstractStep):

    step_name = 'PL'

    required_parameters = []
    input_files = ['blinks']
    output_files = {'previous_pulse_lengths': '.pkl.gz'}


    def perform(self, **kwargs):
        print('------ EXTRACTING PULSE LENGTHS ------')

        blinks = self.load_file('blinks')
        print(blinks)
        previous_pulse_lengths = pd.Series([np.nan] + [blinks[i+1] - blinks[i] for i in range(len(blinks) - 1)], index=blinks.index, name='previous_pulse_length')

        self.save_file(previous_pulse_lengths, 'previous_pulse_lengths')


