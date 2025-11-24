import numpy as np
import torch
import numpy as np
from utils import lab_conv
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import math


def _scaled_free_energy(logits, temperature, mask=None):
    """Helper to compute -T * logsumexp(logits/T) with optional masking."""
    temperature = max(temperature, 1e-6)
    scaled = logits / temperature
    if mask is not None:
        scaled = scaled + mask
    return -temperature * torch.logsumexp(scaled, dim=1)

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
    """Energy-based Active Open-set Annotation (EAOA) sampler.

    Steps:
      1) Extract labeled-set features (for rKNN statistics)
      2) For each unlabeled batch, compute:
         - AU: from C-way head via energy difference between best-vs-rest logits
         - EU: energy from (C+1) head; or energy over known classes + sigmoid of unknown logit
      3) Fit two 2-component GMMs on AU and EU to get info_prob (informativeness) and
         clean_prob (closed-set likelihood)
      4) rKNN energy: use top-K neighbors in labeled features to compute an additional
         energy that stabilizes EU; combine with clean_prob
      5) Per-class selection: for each predicted class, keep top (num_per_class*k1)
         by clean_prob, then sort by info_prob and choose num_per_class
      6) Return selected indices and log precision/recall
    """
    model.eval()  # Set detector model (C+1) to evaluation mode
    model_ID.eval()  # Set classifier model (C) to evaluation mode

    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    # Choose which train loader to use for building rKNN statistics
    if trainloader_ID_w_OOD == None:
        eval_trainloader = trainloader_ID  # Use C-way loader if C+1 loader not available
    else:
        eval_trainloader = trainloader_ID_w_OOD  # Prefer C+1 loader for better rKNN stats

    # STEP 1: Extract labeled-set features for rKNN statistics. neighborhood-based energy metric to provide a stable estimate of the Epistemic Uncertainty (EU) and the closed-set likelihood
    labelArr_all = []  # Store all labeled sample labels
    feat_all = torch.zeros([1, 128], device=device)  # Initialize feature tensor (will remove dummy row)
    
    for batch_idx, (data, labels) in enumerate(eval_trainloader):
        labels = lab_conv(knownclass, labels)  # Map labels to [0..C-1] or C for unknown
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, features = model(data)  # Get 128-d embeddings from detector model
        labelArr_all += list(np.array(labels.cpu().data))
        feat_all = torch.cat([feat_all, features.data],0)  # Concatenate features
    
    feat_all = feat_all[1:]  # Remove dummy first row
    feat_all = F.normalize(feat_all, dim=1)  # L2 normalize for cosine similarity

    # STEP 2: Initialize arrays to store unlabeled pool data
    queryIndex = []  # Store unlabeled sample indices
    labelArr = []    # Store true mapped labels
    predArr = []     # Store predicted classes
    uncertaintyArr = []  # Store epistemic uncertainty (EU) values
    aleatoricUncArr = []  # Store aleatoric uncertainty (AU) values
    precision, recall = 0, 0
    feat_unlab = torch.zeros([1, 128], device=device)  # Store unlabeled embeddings

    # Handle case where unlabeled loader doesn't provide indices directly
    sampler_indices = None
    try:
        if hasattr(unlabeledloader, 'sampler') and hasattr(unlabeledloader.sampler, 'indices'):
            sampler_indices = list(unlabeledloader.sampler.indices)  # Extract indices from sampler
    except Exception:
        sampler_indices = None
    sampler_offset = 0  # Track position in sampler indices

    for batch_idx, batch in enumerate(unlabeledloader):
        # Support either (index, (data, labels)) or (data, labels)
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and not isinstance(batch[0], (list, tuple)):
            data, labels = batch
            if sampler_indices is not None:
                batch_size = labels.shape[0] if hasattr(labels, 'shape') else len(labels)
                index = sampler_indices[sampler_offset:sampler_offset+batch_size]
                sampler_offset += batch_size
            else:
                # Fallback: create dummy per-batch indices (not ideal, but avoids crashes)
                index = list(range(sampler_offset, sampler_offset + (labels.shape[0] if hasattr(labels, 'shape') else len(labels))))
                sampler_offset += (labels.shape[0] if hasattr(labels, 'shape') else len(labels))
        else:
            index, (data, labels) = batch
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data = data.cuda()
        outputs, features = model(data)
        outputs_ID, _ = model_ID(data)

        #a Get predictions from C-way classifier
        softprobs = torch.softmax(outputs_ID, dim=1)
        _, pred = torch.max(softprobs, 1)  # Get predicted class
        
        # Create masks for AU computation: one-hot for predicted class, bias mask
        if use_gpu:
            predTargets = torch.zeros_like(outputs_ID).cuda().scatter_(1, pred.view(-1,1), 1)
            bias = torch.zeros_like(outputs_ID).cuda().scatter_(1, pred.view(-1,1), -1e5) 
        else:
            predTargets = torch.zeros_like(outputs_ID).scatter_(1, pred.view(-1,1), 1)
            bias = torch.zeros_like(outputs_ID).scatter_(1, pred.view(-1,1), -1e5) 
        
        #b ALEATORIC UNCERTAINTY (AU): Energy difference between best vs rest
        # High AU = model is uncertain between predicted class and others
        aleatoricUnc = -torch.logsumexp(outputs_ID, dim=1) + torch.logsumexp(outputs_ID*(1-predTargets)+bias, dim=1)

        # EPISTEMIC UNCERTAINTY (EU): Energy from detector head
        # Low EU = sample is likely from known classes (closed-set)
        if outputs.shape[1] == len(knownclass):
            energy = -torch.logsumexp(outputs, dim=1)  # Energy over all C classes
        else:
            # Energy over known classes + calibrated unknown logit
            energy = -torch.logsumexp(outputs[:,:-1], dim=1) + torch.log(1+torch.exp(outputs[:,-1]))
        Uncertainty = energy

        # Ensure list of ints
        queryIndex += list(np.array(index, dtype=np.int64))
        labelArr += list(np.array(labels.cpu().data))
        predArr += list(np.array(pred.cpu().data))
        uncertaintyArr += list(Uncertainty.cpu().detach().numpy())
        aleatoricUncArr += list(aleatoricUnc.cpu().detach().numpy())
        feat_unlab = torch.cat([feat_unlab, features.data],0)
    feat_unlab = feat_unlab[1:]
    feat_unlab = F.normalize(feat_unlab, dim=1)

    # STEP 3: Fit GMM on AU to get informativeness probability
    aleatoricUncArr = np.asarray(aleatoricUncArr)
    input_aleatoricUncArr = (aleatoricUncArr-aleatoricUncArr.min())/(aleatoricUncArr.max()-aleatoricUncArr.min())  # Normalize to [0,1]
    input_aleatoricUncArr = input_aleatoricUncArr.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_aleatoricUncArr)
    prob = gmm.predict_proba(input_aleatoricUncArr) 
    # info_prob: probability of HIGH AU component (more informative samples)
    info_prob = prob[:,gmm.means_.argmax()] 

    # STEP 4: Fit GMM on EU to get closed-set likelihood probability
    uncertaintyArr = np.asarray(uncertaintyArr)
    input_uncertaintyArr = (uncertaintyArr-uncertaintyArr.min())/(uncertaintyArr.max()-uncertaintyArr.min())  # Normalize to [0,1]
    input_uncertaintyArr = input_uncertaintyArr.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_uncertaintyArr)
    prob = gmm.predict_proba(input_uncertaintyArr) 
    # clean_prob: probability of LOW EU component (more likely closed-set/known classes)
    clean_prob = prob[:,gmm.means_.argmin()] 

    # STEP 5: rKNN energy computation to stabilize EU , neighborhood-based energy metric
    # Compute cosine similarities between labeled and unlabeled features
    dists_all = torch.mm(feat_all, feat_unlab.t())  # Cosine similarity matrix
    _, top_k_index = dists_all.topk(250, dim=1, largest=True, sorted=True)  # Top-250 neighbors for each labeled sample
    dists_all, top_k_index = dists_all.cpu(), top_k_index.cpu()
    rknn_logits = torch.zeros(feat_unlab.shape[0], len(knownclass)+1, dtype=torch.long)  # Count matrix
    
    label_tensor = torch.as_tensor(labelArr_all, dtype=torch.long, device=top_k_index.device)
    if trainloader_ID_w_OOD != None:
        class_count = [0. for i in range(len(knownclass)+1)]
        for i in range(len(knownclass)+1):
            mask = label_tensor == i
            class_count[i] = mask.sum()
            unique_indices, counts = torch.unique(top_k_index[mask, :], return_counts=True)
            rknn_logits[unique_indices,i] = counts
        class_weight = torch.tensor(class_count/np.mean(class_count[:-1])) #.reshape(1, -1)
        rknn_logits = rknn_logits.float()/class_weight
        energy = -torch.logsumexp(rknn_logits[:,:-1], dim=1) + torch.log(1+torch.exp(rknn_logits[:,-1]))
    else:
        class_count = [0. for i in range(len(knownclass)+1)]
        for i in range(len(knownclass)):
            mask = label_tensor == i
            class_count[i] = mask.sum()
            unique_indices, counts = torch.unique(top_k_index[mask, :], return_counts=True)
            rknn_logits[unique_indices,i] = counts
        class_weight = torch.tensor(class_count/np.mean(class_count)) #.reshape(1, -1)
        rknn_logits = rknn_logits.float()/class_weight
        energy = -torch.logsumexp(rknn_logits[:,:-1], dim=1)
    Uncertainty = energy
    uncertaintyArr = list(Uncertainty.cpu().detach().numpy())

    uncertaintyArr = np.asarray(uncertaintyArr)
    input_uncertaintyArr = (uncertaintyArr-uncertaintyArr.min())/(uncertaintyArr.max()-uncertaintyArr.min())
    input_uncertaintyArr = input_uncertaintyArr.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_uncertaintyArr)
    prob = gmm.predict_proba(input_uncertaintyArr) 
    clean_prob2 = prob[:,gmm.means_.argmin()]

    clean_prob *= clean_prob2

    queryIndex = torch.tensor(queryIndex)
    labelArr = torch.tensor(labelArr)
    predArr = torch.tensor(predArr)
    cleanArr = torch.tensor(clean_prob)
    infoArr = torch.tensor(info_prob)
    num_per_class = int(math.ceil(args.query_batch/args.known_class))

    # STEP 6: Two-stage per-class selection (KEY EAOA LOGIC)
    sel_queryIndex, sel_queryLabelArr = [], []
    for i in range(args.known_class):  # For each predicted class. Groups candidates by their predicted class from the C-way classifier.
        # Get samples predicted as class i
        temp_queryIndex = queryIndex[predArr == i]
        temp_labelArr = labelArr[predArr == i]
        temp_cleanArr = cleanArr[predArr == i]  # LOW EU probabilities
        temp_infoArr = infoArr[predArr == i]    # HIGH AU probabilities

        # STAGE 1: Filter by LOW EU (high clean_prob) - select candidates likely to be closed-set
        _, indices = torch.topk(temp_cleanArr.view(-1), num_per_class*int(args.k1))
        temp_queryIndex = temp_queryIndex[indices]
        temp_labelArr = temp_labelArr[indices]
        temp_infoArr = temp_infoArr[indices]

        # STAGE 2: Among candidates, rank by HIGH AU (high info_prob) - select most informative
        tmp_data = np.vstack((temp_queryIndex.numpy(), temp_labelArr.numpy()))
        tmp_data = np.vstack((tmp_data, temp_infoArr.numpy())).T
        tmp_data = tmp_data[(-tmp_data[:,2]).argsort()]  # Sort by info_prob DESCENDING
        tmp_data = tmp_data.T
        sel_queryIndex += list(tmp_data[0][:num_per_class])  # Take top num_per_class
        sel_queryLabelArr += list(tmp_data[1][:num_per_class]) 

    # Final processing and metrics calculation
    queryIndex = np.asarray(sel_queryIndex).astype(np.int32)
    queryLabelArr = np.asarray(sel_queryLabelArr)

    # Random permutation for tie-breaking and diversity
    idx = torch.randperm(len(sel_queryIndex))
    queryIndex = queryIndex[idx][:args.query_batch]  # Cut to exact budget
    queryLabelArr = queryLabelArr[idx][:args.query_batch]

    labelArr = labelArr.numpy()

    # Calculate precision: fraction of selected samples that are truly known
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    # Calculate recall: fraction of all known samples in pool that were selected
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    
    # Return valid (known) and invalid (unknown) indices, plus precision/recall
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def ts_eaoa_sampling(args, unlabeledloader, Len_labeled_ind_train, model, model_ID, knownclass, use_gpu):
    """Implementation of Temperature-Scaled EAOA sampling described in TS-EAOA paper."""
    model.eval()
    model_ID.eval()
    temperature = args.temperature

    sampler_indices = None
    try:
        if hasattr(unlabeledloader, 'sampler') and hasattr(unlabeledloader.sampler, 'indices'):
            sampler_indices = list(unlabeledloader.sampler.indices)
    except Exception:
        sampler_indices = None
    sampler_offset = 0

    idx_list, label_list = [], []
    seu_scores, sau_scores = [], []

    for batch in unlabeledloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and not isinstance(batch[0], (list, tuple)):
            data, labels = batch
            batch_size = labels.shape[0] if hasattr(labels, 'shape') else len(labels)
            if sampler_indices is not None:
                index = sampler_indices[sampler_offset:sampler_offset+batch_size]
            else:
                index = list(range(sampler_offset, sampler_offset + batch_size))
            sampler_offset += batch_size
        else:
            index, (data, labels) = batch
            batch_size = data.shape[0] if hasattr(data, 'shape') else len(data)
            sampler_offset += batch_size

        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data = data.cuda()
        det_logits, _ = model(data)
        cls_logits, _ = model_ID(data)

        known_logits = det_logits[:, :len(knownclass)]
        if det_logits.shape[1] > len(knownclass):
            unknown_logit = det_logits[:, len(knownclass)]
            e_unk = -unknown_logit
        else:
            e_unk = torch.zeros_like(known_logits[:, 0])
        e_known = _scaled_free_energy(known_logits, temperature)
        seu = e_known - e_unk

        e_total = _scaled_free_energy(cls_logits, temperature)
        pred = cls_logits.argmax(dim=1)
        if use_gpu:
            mask = torch.zeros_like(cls_logits).cuda()
        else:
            mask = torch.zeros_like(cls_logits)
        mask.scatter_(1, pred.view(-1, 1), -1e9)
        e_secondary = _scaled_free_energy(cls_logits, temperature, mask=mask)
        sau = e_total - e_secondary

        idx_list += list(np.array(index, dtype=np.int64))
        label_list += list(np.array(labels.cpu().data))
        seu_scores += list(seu.cpu().detach().numpy())
        sau_scores += list(sau.cpu().detach().numpy())

    if len(idx_list) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0., 0.

    candidate_size = max(args.query_batch, int(math.ceil(args.k1 * args.query_batch)))
    candidate_size = min(candidate_size, len(idx_list))

    idx_tensor = torch.tensor(idx_list)
    label_tensor = torch.tensor(label_list)
    seu_tensor = torch.tensor(seu_scores)
    sau_tensor = torch.tensor(sau_scores)

    candidate_order = torch.argsort(seu_tensor)[:candidate_size]
    candidate_au = sau_tensor[candidate_order]
    au_order = torch.argsort(candidate_au, descending=True)
    final_order = candidate_order[au_order][:min(args.query_batch, len(candidate_order))]

    selected_indices = idx_tensor[final_order].numpy().astype(np.int32)
    selected_labels = label_tensor[final_order].numpy()

    precision = 0.
    if len(selected_labels) > 0:
        precision = len(np.where(selected_labels < args.known_class)[0]) / len(selected_labels)
    recall = (len(np.where(selected_labels < args.known_class)[0]) + Len_labeled_ind_train) / \
             (len(np.where(label_tensor.numpy() < args.known_class)[0]) + Len_labeled_ind_train)

    known_mask = selected_labels < args.known_class
    return selected_indices[known_mask], selected_indices[~known_mask], precision, recall
