from sklearn.neighbors import NearestNeighbors
import numpy as np

from core.step_manager import AbstractStep
from utils.math_utils import plogp
import pandas as pd


def compute_mi_ksglike(X: pd.Series, Y: pd.Series, k: int = 20):
    categories = np.unique(X)
    labels = np.array(range(len(X)))
    Y_arr = np.array([y for y in Y])
    X_arr = np.array([x for x in X])
    print(Y_arr)
    print(Y_arr.shape)
    knn_marginal = NearestNeighbors(n_neighbors=k)
    knn_marginal.fit(Y_arr)
    # nearest_neighbors_marginal = NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(Y)
    # nearest_neighbors_joint = []
    

    eps = 1e-6

    conditional_entropy_partial = 0
    input_entropy_partial = 0

    for category in categories:
        Xi = X[X == category]
        nx = len(Xi)
        labels_i = labels[X == category]

        Yi = Y_arr[X == category]
        dists, neighs = NearestNeighbors(n_neighbors=k).fit(Yi).kneighbors(Yi)
        kth_dists = [ds[-1] + eps for ds in dists]
        for x, y, dist, i in zip(Xi, Yi, kth_dists, labels_i):
            dists_i, neighs_i = knn_marginal.radius_neighbors([y], radius=dist)
            ny = len(neighs_i[0])
            conditional_entropy_partial += -np.log2(ny) + np.log2(k)

        input_entropy_partial += plogp(len(Xi))
    input_entropy = (input_entropy_partial - plogp(len(X))) / len(X)
    conditional_entropy = conditional_entropy_partial / len(X)
        # nearest_neighbors_joint.append()
    print(input_entropy, conditional_entropy)
    return input_entropy - conditional_entropy

   

class Step(AbstractStep):

    step_name = 'MIkd'

    required_parameters = ['target_position', 'nearest_neighbor_k', 'entropy_estimation_correction']
    input_files = ['extracted_slices']
    output_files = {'mutual_information': '.pkl.gz'}#, 'prediction_probas': '.pkl.gz'}



    def perform(self, **kwargs):
        print('------ESTIMATING MI (DISCRETE)------')

        tpos = kwargs['target_position']
        nearest_neighbor_k = kwargs['nearest_neighbor_k']
        entropy_estimation_correction = kwargs['entropy_estimation_correction']

        slices: pd.DataFrame = self.load_file('extracted_slices')
        slices = slices

        print('Reformating data', end='... ', flush=True)
        # data = list(zip(1 * (slices['target'].eq(tpos)), slices['flat_data']))#[(1 * (row['target'] == tpos), np.array(row['flat_data'])) for idx,row in slices.iterrows()]

        print('Computing MI', end='... ', flush=True)
        mutual_information = compute_mi_ksglike(slices['target'].eq(tpos), pd.Series([np.array(y) for y in slices['flat_data']], index=slices.index), nearest_neighbor_k, entropy_estimation_correction)
        print('done.')


        self.save_file(mutual_information, 'mutual_information')

