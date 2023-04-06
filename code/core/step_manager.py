# step_manager.py
from abc import ABC, abstractmethod
import os
import json
import pickle, gzip

from typing import Dict, Any
import hashlib
import json
from pathlib import Path

from datetime import datetime

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def printor(x):
    print(x)
    return x


class Chain:
    def __init__(self, parameters, cache_directory=Path('.'), working_directory=Path('.')): 
        self.parameters = parameters 
        self.module = None
        self.files : dict[str, ChainFile]= {} 
        self.important_parameters = {}
        self.steps = []
        self.step_dict = {}

        cache_directory = Path(cache_directory)
        working_directory = Path(working_directory)
        self.cache_directory = cache_directory
        self.working_directory = working_directory.absolute()
        working_directory.mkdir(parents=True, exist_ok=True)
        os.chdir(str(working_directory))
      
        self.cache_directory_path = cache_directory.absolute()
        print(self.cache_directory)
        cache_directory.mkdir(parents=True, exist_ok=True)
        # for i in range(len(self.cache_directory.split('/'))):
        #     if i == 0 and self.cache_directory[0] == '/': continue
        #     if not os.path.isdir(os.sep.join(self.cache_directory.split('/')[:(i+1)])):
        #         os.mkdir(os.sep.join(self.cache_directory.split('/')[:(i+1)]))

        print(datetime.now())
        print(parameters)
        print(os.curdir)

    def step(self, module, useCache='if_up_to_date'):
        assert useCache in ['if_present', 'if_up_to_date', 'never']

        step : AbstractStep = module.Step(self)
        step.last_modified = os.path.getmtime(module.__file__) or datetime.now().timestamp()
        step.full_step_name = module.__name__ or self.step_name
        
        if useCache == 'if_present':
            step.perform(**self.parameters) if not self.files_exist(step.output_files) else print(f'Using cached output from \'{ step.full_step_name}\'')
        elif useCache == 'if_up_to_date':
            step.perform(**self.parameters) if not self.are_output_files_up_to_date(step) else print(f'Using cached output from \'{ step.full_step_name }\'')
        else:
            step.perform(**self.parameters)
        return self
    
    def get_file_abspath_by_type(self, file_type):
        return self.cache_directory_path / self.files[file_type].filename

    def files_exist(self, filetypes):
        return all([filename.exists() or print('File does not exist:', filename) for filename in [self.get_file_abspath_by_type(filetype) for filetype in filetypes]])

    def are_output_files_up_to_date(self, step):
        
        assert self.files_exist(step.input_files), f'{step.full_step_name}: not all input files exist: {str(step.input_files)}\nAvailable input files are {str(self.files.keys())}'

        last_input_file_modified = max([0.] + [
            os.path.getmtime(filename)
            for filetype in step.input_files
            for filename in [self.get_file_abspath_by_type(filetype)]
            ])

        return all([
            filename.exists()
            and os.path.getmtime(filename) >= step.last_modified
            and os.path.getmtime(filename) >= last_input_file_modified
            for filetype in step.output_files
            for filename in [self.get_file_abspath_by_type(filetype)]
            ])

    def load_file(self, filetype, verbose=True):
        if verbose: print(f"Loading {filetype} from '{self.get_file_abspath_by_type(filetype)}'", end='... ', flush=True)
        with gzip.open(self.get_file_abspath_by_type(filetype), 'rb') as pof:
            output = pickle.load(pof)
        if verbose: print('done', flush=True)
        return output

    def save_file(self, data, filetype, verbose=True):
        if verbose: print(f"Saving {filetype} to '{self.get_file_abspath_by_type(filetype)}'", end='... ', flush=True)
        with gzip.open(self.get_file_abspath_by_type(filetype), 'wb') as pof:
            pickle.dump(data, pof)
        if verbose: print('done', flush=True)
        return self

    def add_file(self, as_filetype, file, extension='.pkl.gz'):
        self.parameters = {**self.parameters, '_file_for_' + as_filetype : file}
        FileAdder(self, as_filetypes={as_filetype: extension}, external_files={as_filetype: os.path.relpath(file, self.cache_directory_path)})
        return self

    def add_files(self, as_filetypes, filenames, extension='.pkl.gz'):
        self.parameters = {**self.parameters, **{'_file_for_' + as_filetype: filename for as_filetype,filename in zip(as_filetypes, filenames)}}
        FileAdder(self, 
            as_filetypes={as_filetype: extension for as_filetype in as_filetypes}, 
            external_files={as_filetype: os.path.relpath(filename, self.cache_directory_path) for as_filetype,filename in zip(as_filetypes, filenames)})
        return self

    def add_output_file(self, filetype, predecessors = [], extension='.pkl.gz'):
        FileAdder(self, as_filetypes={filetype: extension}, predecessors=predecessors)
        return self
    

def escape_arguments(arguments, required_keys=None):

    keys = required_keys or arguments.keys()
    # escaper = dict(((ord(char), '%') for char in string.punctuation + ' '))
    return dict_hash({key: str(arguments[key]) for key in keys})#'--'.join([key[:5] + '-' + str(arguments[key]).translate(escaper) for key in keys])

def create_filename(arguments, required_keys, step_name, file_type='', extension=''):
    return str(escape_arguments(arguments, required_keys)) + '--' + step_name + ('--' + file_type if file_type else '') + extension

def load_file(filename, filetype='', verbose=True):
    if verbose: print(f'Loading {filetype} from \'{filename}\'', end='... ', flush=True)
    with gzip.open(filename, 'rb') as pof:
        content = pickle.load(pof)  
    if verbose: print('done', flush=True)
    return content

def save_file(data, filename, filetype='', verbose=True):
    if verbose: print(f'Saving {filetype} to \'{filename}\'', end='... ', flush=True)
    with gzip.open(filename, 'wb') as pof:
        pickle.dump(data, pof)  
    if verbose: print('done', flush=True)


class AbstractStep(ABC):

    def __init__(self, chain: Chain, custom_filenames = {}) -> None:
        
        assert not self.input_files is None, f'Step \'{self.full_step_name}\': Input files must be specified'
        assert not self.output_files is None, f'Step \'{self.full_step_name}\': Output files must be specified'
        assert not self.required_parameters is None, f'Step \'{self.full_step_name}\': Required must be specified'
        assert not self.step_name is None, f'Step \'{self.full_step_name}\': Required must be specified'



        for filetype in self.output_files:
            assert filetype not in chain.files, f'Step {self.step_name} would duplicate file {filetype} in the chain!'
        for filetype in self.input_files:
            assert filetype in chain.files, f'Step {self.step_name} requires file {filetype}, which is not present in the chain!'

        if self.step_name in chain.steps:
            i=2
            while (self.step_name + str(i) in chain.steps): i+=1
            print(f"WARNING: step {self.step_name} already in chain. Changing step name to {self.step_name + str(i)}.")
            self.step_name = self.step_name + str(i)

        super().__init__()
        self.chain = chain
        chain.steps.append(self.step_name)
        chain.step_dict = {**chain.step_dict,  self.step_name: self}
        self.predecessors = set.union(set({}), *(chain.files[file].predecessors for file in self.input_files))   

        self.important_parameters = set(self.required_parameters).union(*(chain.step_dict[chain.files[file].creator].important_parameters for file in self.input_files))
        chain.important_parameters = {**chain.important_parameters, self.step_name: self.important_parameters}


        chain.files = {**chain.files, 
            **{ 
                filetype: ChainFile(
                    creator = self.step_name,
                    previous_steps = chain.steps.copy(),
                    predecessors =  self.predecessors,
                    parameters = self.important_parameters.copy(), 
                    extension = self.output_files[filetype],
                    filename = custom_filenames[filetype] if filetype in custom_filenames else  create_filename(chain.parameters, self.important_parameters, '-'.join(chain.steps), filetype, self.output_files[filetype]) 
                )
                for filetype in self.output_files}
            }


    @abstractmethod
    def perform(self, **kwargs):
        pass

    def load_file(self, filetype, verbose=True, ):
        assert filetype in self.input_files, f"Step {self.step_name} is trying to load file {filetype} which is not listed in input files"
        return self.chain.load_file(filetype=filetype, verbose=verbose)
    def save_file(self, data, filetype, verbose=True):
        assert filetype in self.output_files, f"Step {self.step_name} is trying to save file {filetype} which is not listed in output files"
        self.chain.save_file(data=data, filetype=filetype, verbose=verbose)


class ChainFile():
    
    def __init__(self, creator, previous_steps, predecessors, parameters, extension, filename) -> None:
        self.creator = creator
        self.previous_steps = previous_steps
        self.predecessors = predecessors
        self.parameters = parameters
        self.extension = extension
        self.filename = filename


class FileAdder(AbstractStep):

    step_name = '_FA_'

    def __init__(self, chain: Chain, as_filetypes=[], external_files = {}, additional_parameters=[], predecessors=[]) -> None:
        self.input_files = list(predecessors)
        self.output_files = as_filetypes
        self.required_parameters = ['_file_for_' + filetype for filetype in external_files] + additional_parameters
        super().__init__(chain, custom_filenames=external_files)

    def perform(self, **kwargs):
        pass


class FileMerger(AbstractStep):
    pass

