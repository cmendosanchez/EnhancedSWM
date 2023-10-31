from dipy.tracking.distances import bundles_distances_mdf
import read_write_bundle as bt
import numpy as np
from operator import itemgetter 
import os 
from subprocess import call
import argparse
from scipy.spatial.distance import cdist
import math
from joblib import Parallel, delayed
import multiprocessing
from collections import defaultdict
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
from dipy.tracking.streamline import set_number_of_points
from scipy.spatial import distance


def distance_matrix_Dend(bun_in):
    '''
    Compute the Dend distance of each fiber in a bundle to every other fiber.
    Returns the distance matrix based on the SSPD distance.
    '''
    end_p1 = []
    for fib in bun_in:
        end_p1.append([fib[0], fib[-1]])

    return  bundles_distances_mdf(end_p1, end_p1)




def distance_matrix_SSPD(bun_in):
    '''
    Compute the SSPD distance of each fiber in a bundle to every other fiber.
    Returns the distance matrix based on the SSPD distance.
    '''
    call(['./SSPD3D 21 '+bun_in +' sspd.txt'],shell=True)
    fasc = bt.read_bundle(bun_in)
    matrix_dist = np.zeros((len(fasc),len(fasc)))
    with open('sspd.txt') as f:
        lines = f.readlines()
    for i,l in enumerate(lines):
        l1 = l.split()
        for j,v in enumerate(l1):
            matrix_dist[i][j] = float(v)
    
    return matrix_dist

def matrix_dist_filtering(matrix_dist,qth_thres,dist_thres,streamlines):
    '''
    Filtering of streamlines based on an input distance matrix.
    Returns the filtered set of streamlines.
    '''
    matrix_dist = np.where(matrix_dist<dist_thres,1,0)
    np.fill_diagonal(matrix_dist, 0)
    degree = np.sum(matrix_dist,axis=1)
    qth_thres = np.percentile(degree,qth_thres)
    filtered_idx = np.where(degree>=qth_thres)[0]
    noise_idx = np.where(degree<qth_thres)[0]
    filtered_bundle =  list(itemgetter(*filtered_idx)(streamlines))
    return filtered_bundle,noise_idx


def Consistency(bun_in,dist_matrix,sigma,k):
    '''
    Returns the Fiber Consistency of each fiber in a bundle.
    '''

    bundle = bt.read_bundle(bun_in)
    sl_consistency = []
    features_matrix = np.empty((len(bundle),21))
    fiber_mean_consistency = []

    for j in range(0,len(bundle)):
        A = dist_matrix[j]
        neighbors_idx = np.argsort(A)[:k+1]
        n_fibers = []
        for i in neighbors_idx:
            if i!=j:
                n_fibers.append(bundle[i])

        sum_list = np.zeros((21,k))
        #print(len(n_fibers))
        for idx,n_f in enumerate(n_fibers):
            distances = cdist(bundle[j],n_f,'euclidean')
            min_dist = np.amin(distances,axis=1)
            topho_dist = np.exp(-pow(min_dist,2)/pow(sigma,2))
            sum_list[:,idx] = topho_dist
        #print(sum_list,sum_list.shape)
        sum_list = np.sum(sum_list,axis=1)
        fiber_mean_consistency.append(np.mean(sum_list))
    fiber_mean_consistency = np.array(fiber_mean_consistency)
    return fiber_mean_consistency

def ConsistencyValsFiltering(bundle,fiber_mean_consistency,qth_thres):
    '''
    Filtering of streamlines based on Fiber Consistency.
    Returns the filtered set of streamlines.
    '''
    qth = np.percentile(fiber_mean_consistency,qth_thres)
    filtered_idx = np.where(fiber_mean_consistency>qth)[0]
    noise_idx = np.where(fiber_mean_consistency<=qth)[0]
    filtered_bundle= list(itemgetter(*filtered_idx)(bundle))
    return filtered_bundle,noise_idx





def ConvexHullFiltering(bundle_in,q_th,k,npoints):
    '''
    Filtering of streamlines based on the Convex Hull.
    Returns the filtered set of streamlines.
    '''
    fibras = bt.read_bundle(bundle_in)
    fibras_init = len(fibras)
    my_dict = {}
    for idx,sl in enumerate(fibras):
        my_dict[sl.tobytes()] = idx


    deleted_fibers = []
    noise_idx = []
    discarded_percentaje = 0
    while(discarded_percentaje<q_th):
        pts = np.concatenate(fibras)
        hull = ConvexHull(pts)
        vert_idx = hull.vertices
        kdt = KDTree(pts, metric='euclidean')
        fibs_DA = []
        fibras_CH_idx = []
        for idx,v in enumerate(vert_idx):
            fibras_CH_idx.append(int(v/npoints))
        fibras_CH_idx = list(set(fibras_CH_idx))
        fibras_CH = [ fibras[i] for i in fibras_CH_idx]

        for fib in fibras_CH:
            dist, ind = kdt.query(fib, k=k) 
            each_p_md = []
            for d in dist:
                each_p_md.append(np.mean(d))
            fibs_DA.append(np.mean(each_p_md))

        most_atipic = np.where(fibs_DA>np.mean(fibs_DA)+np.std(fibs_DA))[0]
        if len(most_atipic) == 0:
            break

        noise_idx = [ fibras_CH_idx[i] for i in most_atipic]
        for idx in sorted(noise_idx, reverse=True):
            deleted_fibers.append(fibras[idx])
            del fibras[idx]

        discarded_percentaje = ((fibras_init-len(fibras))/fibras_init)*100

    noise_idx = []
    for sl in deleted_fibers:
        noise_idx.append(my_dict[sl.tobytes()])

    return fibras,noise_idx

