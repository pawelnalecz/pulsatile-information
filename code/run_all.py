import subprocess as sp
from pathlib import Path
from time import sleep

my_dir = Path(__file__).parent

workers = 9

scripts = [
    'fig1CD.py',
    'fig1EF.py',
    'fig1G.py',
    'fig2ACE.py',
    'fig2BDF.py',
    'fig3ABC_4AB.py',
    'fig3DE.py',
    'figS3A.py',
    'figS3BCD.py',
]

def run_script(script):
    print(f'Running {script}')
    return sp.Popen(['python', str(my_dir / 'figures' / script)], shell=True)

def get_run_script(script):
    return lambda: run_script(script)

jobs = []

for script in scripts:
    jobs.append(get_run_script(script))



workers_free = workers
jobs_iter = iter(jobs)
jobs_running = []

while True:
    for it_job, job in enumerate(jobs_running):
        if job.poll() is not None:
             workers_free += 1
             jobs_running.pop(it_job)
             break

    if workers_free > 0:
        workers_free -= 1
        next_job = next(jobs_iter, None)
        if next_job is not None:
        	jobs_running.append(next_job())
        else: 
            break
    else:
        sleep(1)

for job in jobs_running:
    job.wait()

print('ALL JOBS COMPLETED')
    
