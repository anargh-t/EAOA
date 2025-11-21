import numpy as np
import torch
from utils import lab_conv, scaled_free_energy
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import math


def _fit_gmm_probability(values, pick='high'):
    """
    Fit a two-component GMM to values and return probability of the selected component.

    Args:
        values (array-like): 1D array of raw scores.
        pick (str): 'high' to select component with higher mean, 'low' for lower mean.

    Returns:
        np.ndarray of probabilities aligned with the input values.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    if values.size < 2 or np.isclose(values.max(), values.min()):
        return np.full(values.shape[0], 0.5, dtype=np.float32)

    normalized = ((values - values.min()) / (values.max() - values.min())).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    try:
        gmm.fit(normalized)
        probs = gmm.predict_proba(normalized)
        component = gmm.means_.argmax() if pick == 'high' else gmm.means_.argmin()
        return probs[:, component]
    except ValueError:
        return np.full(values.shape[0], 0.5, dtype=np.float32)

def random_sampling(args, unlabeledloader, Len_labeled_ind_train, model, knownclass):
    """Baseline random sampler: pick args.query_batch indices uniformly.

    Returns:
        valid_indices: indices predicted to be known (for logging precision)
        invalid_indices: indices predicted to be unknown
        precision: fraction of picked indices truly known (w.r.t. mapped labels)
        recall: fraction of all known in unlabeled pool that were selected
    """
    model.eval()  # Set model to evaluation mode (not used for scoring in random sampling)
    queryIndex = []  # Store all unlabeled indices
    labelArr = []    # Store corresponding mapped labels
    precision, recall = 0, 0
    
    # Collect all unlabeled samples and their labels
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)  # Map original labels to [0..C-1] or C for unknown
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    # Shuffle all samples uniformly (random selection)
    tmp_data = np.vstack((queryIndex, labelArr)).T  # Stack indices and labels
    np.random.shuffle(tmp_data)  # Random shuffle
    tmp_data = tmp_data.T
    queryIndex = tmp_data[0][:args.query_batch]  # Select first query_batch samples
    labelArr = tmp_data[1]
    queryLabelArr = tmp_data[1][:args.query_batch]
    
    # Calculate precision: fraction of selected samples that are truly known
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    # Calculate recall: fraction of all known samples in pool that were selected
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    
    # Return valid (known) and invalid (unknown) indices, plus precision/recall
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def eaoa_sampling(args, unlabeledloader, Len_labeled_ind_train, model, model_ID, knownclass, use_gpu, query, trainloader_ID, trainloader_ID_w_OOD):
    """Energy-based Active Open-set Annotation (EAOA) sampler with temperature-scaled energies."""
    model.eval()
    model_ID.eval()

    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    temperature = getattr(args, "temperature", 1.0)

    eval_trainloader = trainloader_ID if trainloader_ID_w_OOD is None else trainloader_ID_w_OOD

    # STEP 1: Extract labeled-set features for rKNN statistics
    labelArr_all = []
    feat_all = torch.zeros([1, 128], device=device)
    for data, labels in eval_trainloader:
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, features = model(data)
        labelArr_all += list(np.array(labels.cpu().data))
        feat_all = torch.cat([feat_all, features.data], 0)
    feat_all = F.normalize(feat_all[1:], dim=1)

    # STEP 2: Gather unlabeled statistics
    queryIndex, labelArr = [], []
    eu_model_values, au_values = [], []
    feat_unlab = torch.zeros([1, 128], device=device)

    sampler_indices = None
    try:
        if hasattr(unlabeledloader, 'sampler') and hasattr(unlabeledloader.sampler, 'indices'):
            sampler_indices = list(unlabeledloader.sampler.indices)
    except Exception:
        sampler_indices = None
    sampler_offset = 0

    for batch in unlabeledloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and not isinstance(batch[0], (list, tuple)):
            data, labels = batch
            batch_size = labels.shape[0] if hasattr(labels, 'shape') else len(labels)
            if sampler_indices is not None:
                index = sampler_indices[sampler_offset:sampler_offset + batch_size]
            else:
                index = list(range(sampler_offset, sampler_offset + batch_size))
            sampler_offset += batch_size
        else:
            index, (data, labels) = batch
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data = data.cuda()
        outputs, features = model(data)
        outputs_ID, _ = model_ID(data)

        _, pred = torch.max(outputs_ID, 1)
        bias = torch.zeros_like(outputs_ID).scatter_(1, pred.view(-1, 1), -1e5)

        total_energy = scaled_free_energy(outputs_ID, temperature)
        secondary_energy = -temperature * torch.logsumexp((outputs_ID + bias) / temperature, dim=1)
        aleatoricUnc = total_energy - secondary_energy

        energy_known = scaled_free_energy(outputs[:, :-1], temperature)
        energy_unknown = scaled_free_energy(outputs[:, -1:].contiguous(), temperature)
        eu_model = energy_known - energy_unknown

        queryIndex += list(np.array(index, dtype=np.int64))
        labelArr += list(np.array(labels.cpu().data))
        eu_model_values += list(eu_model.detach().cpu().numpy())
        au_values += list(aleatoricUnc.detach().cpu().numpy())
        feat_unlab = torch.cat([feat_unlab, features.data], 0)

    feat_unlab = F.normalize(feat_unlab[1:], dim=1)

    # STEP 3: Data-driven EU via rKNN counts
    dists_all = torch.mm(feat_all, feat_unlab.t())
    _, top_k_index = dists_all.topk(250, dim=1, largest=True, sorted=True)
    top_k_index = top_k_index.cpu()
    rknn_logits = torch.zeros(feat_unlab.shape[0], len(knownclass) + 1, dtype=torch.long)

    label_tensor = torch.as_tensor(labelArr_all, dtype=torch.long, device=top_k_index.device)
    if trainloader_ID_w_OOD is not None:
        class_count = [0. for _ in range(len(knownclass) + 1)]
        for i in range(len(knownclass) + 1):
            mask = label_tensor == i
            class_count[i] = mask.sum()
            unique_indices, counts = torch.unique(top_k_index[mask, :], return_counts=True)
            rknn_logits[unique_indices, i] = counts
        class_count = np.array(class_count, dtype=np.float32)
        class_weight = torch.tensor(class_count / np.mean(class_count[:-1] if len(class_count) > 1 else class_count), dtype=torch.float32)
        rknn_logits = rknn_logits.float() / class_weight
        data_driven_energy = -torch.logsumexp(rknn_logits[:, :-1], dim=1) + torch.log(1 + torch.exp(rknn_logits[:, -1]))
    else:
        class_count = [0. for _ in range(len(knownclass) + 1)]
        for i in range(len(knownclass)):
            mask = label_tensor == i
            class_count[i] = mask.sum()
            unique_indices, counts = torch.unique(top_k_index[mask, :], return_counts=True)
            rknn_logits[unique_indices, i] = counts
        class_count = np.array(class_count, dtype=np.float32)
        denom = np.mean(class_count if len(knownclass) > 0 else np.array([1.], dtype=np.float32))
        class_weight = torch.tensor(class_count / denom, dtype=torch.float32)
        rknn_logits = rknn_logits.float() / class_weight
        data_driven_energy = -torch.logsumexp(rknn_logits[:, :-1], dim=1)
    eu_data_values = list(data_driven_energy.cpu().detach().numpy())

    # STEP 4: Convert raw scores to probabilities
    sL_prob = _fit_gmm_probability(eu_model_values, pick='high')
    sD_prob = _fit_gmm_probability(eu_data_values, pick='high')
    score_eu = sL_prob * sD_prob
    score_au = _fit_gmm_probability(au_values, pick='high')

    queryIndex = np.asarray(queryIndex)
    labelArr = np.asarray(labelArr)
    score_eu = np.asarray(score_eu)
    score_au = np.asarray(score_au)

    if queryIndex.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), 0., 0.

    candidate_budget = int(math.ceil(args.k1 * args.query_batch))
    candidate_budget = max(args.query_batch, candidate_budget)
    candidate_budget = min(candidate_budget, queryIndex.size)

    candidate_order = np.argsort(score_eu)[:candidate_budget]
    final_budget = min(args.query_batch, candidate_order.size)
    final_rank = candidate_order[np.argsort(-score_au[candidate_order])[:final_budget]]

    queryIndex_selected = queryIndex[final_rank]
    queryLabelArr = labelArr[final_rank]

    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)

    known_mask = queryLabelArr < args.known_class
    return queryIndex_selected[known_mask], queryIndex_selected[~known_mask], precision, recall
