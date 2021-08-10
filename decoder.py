def assign_statistic(
    observables: torch.Tensor, # [time, n_neurons]
    statistic: torch.Tensor = None,
    alpha: float = 1.0,
) -> torch.Tensor:

    time = observables.size(0)
    n_neurons = observables.size(1)

    if statistic is None:
        statistic = torch.zeros((n_neurons), device=observables.device)
        
    # Compute first component for this label.
    _, S, V = torch.pca_lowrank(observables, q=1, center=False)

    '''
    statistic = (alpha*statistic + (1-alpha)*V) # Is there learning in this statistic?
    statistic = statistic / (alpha*alpha-2*alpha) # Normalization since it only represents directions, maybe magnitude should be included.
    '''
    statistic = V
        
    return statistic.squeeze()

# O(time)
def sliding_window_sum(observable, window_size=5):
    cum_spikes = torch.zeros(observable.shape, device=observable.device)

    tmp = observable[range(window_size),:].sum(dim=0)

    offset = int((window_size / 2) - 1)
    for idx in range(observable.size(0)):
        cum_spikes[idx,:] = tmp
        if idx - offset >= 0:
            tmp -= observable[idx - offset,:].int()
        if idx + offset + 1 < cum_spikes.size(0):
            tmp += observable[idx + offset + 1,:].int()
            
    return cum_spikes

def Kmeans_median(
    statistic_record: torch.Tensor,
    labels: torch.Tensor,
    n_labels: int
):
    n_neurons = statistic_record.size(1)

    cluster_ids, cluster_centers = kmeans(
        X=statistic_record, 
        num_clusters=n_labels, 
        distance='euclidean', 
        device=statistic_record.device,
        tqdm_flag = False
    )

    centers_lbls = torch.zeros((n_labels), device=statistic_record.device)
    assigned = []
    
    i = 0
    for i in range(n_labels):
        centers_lbls[i], _ = torch.mode(labels[cluster_ids[i]])
        '''
        if centers_lbls[i] not in assigned:
            assigned.append(centers_lbls[i])
            i += 1
        elif len(labels[cluster_ids[i]].shape) == 0:
            centers_lbls[i] = -1
            i += 1
        else: 
            labels[cluster_ids[i]][indices] = 300 + indices
        print(assigned)
        '''
    return cluster_centers, centers_lbls

def PCA_average(
    statistic_record: torch.Tensor,
    labels: torch.Tensor,
    n_labels: int
):

    n_neurons = statistic_record.size(1)
    statistic_avg = torch.zeros((n_neurons, n_labels), device=statistic_record.device)

    for j in range(n_labels):
        i=1
        # Count the number of samples with this label.
        n_labeled = torch.sum(labels == i).float().item()

        if n_labeled > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(labels == i).view(-1)
            
            # Assign label i to average principal component
            statistic_avg[:,i] = torch.sum(statistic_record[indices], 0) / n_labeled

    return statistic_avg

"""
Copy of kmeans pytorch implementation found in https://github.com/subhadarship/kmeans_pytorch
"""
from functools import partial

import numpy as np
import torch
from tqdm import tqdm


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=False,
        iter_limit=0,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if tqdm_flag:
        print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    elif distance == 'soft_dtw':
        sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
        pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters)
    else:
        if tqdm_flag:
            print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:

        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        tqdm_flag=False
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor) cluster ids
    """
    if tqdm_flag:
        print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    elif distance == 'soft_dtw':
        sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
        pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu'), tqdm_flag=False):
    if tqdm_flag:
        print(f'device is :{device}')
    
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def pairwise_soft_dtw(data1, data2, sdtw=None, device=torch.device('cpu')):
    if sdtw is None:
        raise ValueError('sdtw is None - initialize it with SoftDTW')

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # (batch_size, seq_len, feature_dim=1)
    A = data1.unsqueeze(dim=2)

    # (cluster_size, seq_len, feature_dim=1)
    B = data2.unsqueeze(dim=2)

    distances = []
    for b in B:
        # (1, seq_len, 1)
        b = b.unsqueeze(dim=0)
        A, b = torch.broadcast_tensors(A, b)
        # (batch_size, 1)
        sdtw_distance = sdtw(b, A).view(-1, 1)
        distances.append(sdtw_distance)

    # (batch_size, cluster_size)
    dis = torch.cat(distances, dim=1)
    return dis