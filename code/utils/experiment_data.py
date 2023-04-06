old_experiments = [ 
    'min3_mean20', 
    'min3_mean30', 
    'min3_mean40',
    'min10_optmean',
    'min15_optmean', 
    'min20_optmean', 
    'min20_optmeanb',
    *[f'pseudorandom_pos{pos:02d}_period{period:d}' for period in [10, 15, 20, 30] for pos in range(1, 11) ]
    ]
new_experiments = [ 
    'min3_mean20_new', 
    'min3_mean30_new', 
    'min3_mean40_new',
    'min10_optmean_new',
    'min15_optmean_new', 
    'min20_optmean_new', 
    'min30_optmean_new',
    *[f'pseudorandom_pos{pos:02d}_period{period:d}' for period in [5, 10] for pos in range(1, 11) ]
    ]
repeated_experiments = [ 
    'min3_mean50_newb',
    'min15_optmean_newb', 
    'min30_optmean_newb', 
    *[f'pseudorandom_pos{pos:02d}_period{period:d}_new' for period in [3, 10, 15] for pos in range(1, 11)],
    ]
def is_old_experiment(experiment):
    return experiment in old_experiments
def is_repeated_experiment(experiment):
    return experiment in repeated_experiments

