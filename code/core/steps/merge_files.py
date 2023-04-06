# merging_files.py
import pandas as pd

if __name__ == '__main__': import __init__
from core.step_manager import AbstractStep, Chain

class Step(AbstractStep):
    step_name = '_MF_'
    required_parameters = ['merge_file_labels', 'merge_file_names', 'merge_output_filetype', 'merge_index_name']

    def __init__(self, chain: Chain, custom_filenames={}) -> None:
        assert all(key in chain.parameters.keys() for key in self.required_parameters)

        labels = chain.parameters['merge_file_labels'] or chain.parameters['merge_file_names'] 
        output_filetype = chain.parameters['merge_output_filetype']
        filenames = chain.parameters['merge_file_names']

        chain.add_files([output_filetype + '_' + label for label in labels], filenames, extension='.pkl.gz')
        self.input_files = [output_filetype + '_' + label for label in labels]
        self.output_files = {output_filetype: '.pkl.gz'}
        super().__init__(chain, custom_filenames=custom_filenames)

    def perform(self, **kwargs):
        index_name = kwargs['merge_index_name']
        output_filetype = kwargs['merge_output_filetype']
        index_name = kwargs['merge_index_name']
        labels = kwargs['merge_file_labels'] or kwargs['merge_file_names'] 

        print('-------------------- MERGING FILES -----------------')


        def add_label_to_index(df: pd.DataFrame, label):
            df[index_name]  = label
            old_index_names = df.index.names
            return df.reset_index().set_index([index_name] + old_index_names)

        print('Merging', end='... ', flush=True)
        resulting_file = pd.concat(add_label_to_index(self.load_file(output_filetype + '_' + label), label) for label in labels)
        print('done.', flush=True)
        self.save_file(resulting_file, output_filetype)
 


