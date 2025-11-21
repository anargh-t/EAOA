import os
import argparse
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from resnet import ResNet18
import query_strategies
import datasets
from utils import AverageMeter, get_splits, lab_conv, scaled_free_energy
import torch.nn.functional as F

# Argument parser for configuring the CIFAR-10 active learning run
parser = argparse.ArgumentParser("Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach (CIFAR-10 only)")
# dataset (CIFAR-10 only)
parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['cifar10'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
# Mini-batch size for training and evaluation
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=200)  # epochs per active-learning round
parser.add_argument('--max-query', type=int, default=11)  # number of active-learning rounds
parser.add_argument('--query-batch', type=int, default=1500)  # samples to annotate per round (budget)
parser.add_argument('--stepsize', type=int, default=60)  # LR scheduler step
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")  # LR scheduler gamma
parser.add_argument('--query-strategy', type=str, default='eaoa_sampling', choices=['random', 'eaoa_sampling'])  # strategy to select new labels

# model
parser.add_argument('--model', type=str, default='resnet18')
# misc
parser.add_argument('--eval-freq', type=int, default=200)  # evaluate every N epochs
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
# openset
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--is-mini', type=bool, default=True)
parser.add_argument('--known-class', type=int, default=4)  # number of known classes C
parser.add_argument('--init-percent', type=int, default=16)  # initial labeled percent among known-class data

########
parser.add_argument('--energy-weight', type=float, default=0.01)  # weight for energy-margin loss
parser.add_argument('--m-in', type=float, default=-25)  # margin for in-distribution energy
parser.add_argument('--m-out', type=float, default=-7)  # margin for out-of-distribution energy

parser.add_argument('--k1', type=float, default=5)  # top-k multiplier for candidate set size
parser.add_argument('--a', type=float, default=1)  # k1 adjustment step
parser.add_argument('--z', type=float, default=0.05)  # tolerance around target precision
parser.add_argument('--target_precision', type=float, default=0.6)  # target precision for queried labels
parser.add_argument('--temperature', type=float, default=1000.0, help="Temperature T for scaled energy computations")

    

args = parser.parse_args()

import logging

# Set up per-run logfile (helps reproduce and analyze runs)
logging_filename = args.dataset + "_strategy_" + args.query_strategy + "_known-class_" + str(args.known_class) \
                    + "_energy-weight_" + str(args.energy_weight) + "_m-in_" + str(args.m_in) + "_m-out_" + str(args.m_out) \
                    + "_k1_" + str(args.k1) + "_a_" + str(args.a) + "_z_" + str(args.z) + "_seed_" + str(args.seed) + ".log"

logging.basicConfig(level=logging.INFO,
                    filename=logging_filename,
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

def main():
    """Top-level Active Learning loop for CIFAR-10.

    Workflow per query round:
      1) Build loaders for current labeled/unlabeled/test splits
      2) Initialize two models:
         - model_ID:      C-way classifier (known classes only)
         - model_ID_w_OOD:C+1-way detector (known classes + unknown)
      3) Train both models for args.max_epoch epochs
         - model_ID uses standard cross-entropy
         - model_ID_w_OOD uses cross-entropy + energy-margin loss (in/out)
      4) Evaluate on test
      5) Select new indices via EAOA sampling and update splits
      6) Adapt k1 to hit target labeling precision
    """
    seed = 1
    all_accuracies, all_open_accuracies = [], []
    all_precisions = []
    all_recalls = []
    # Determine which raw CIFAR-10 classes become the C known classes
    knownclass = get_splits(args.dataset, seed, args.known_class)  
    print("Known Classes", knownclass)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # select GPU
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    # Construct loaders with labeled/unlabeled split for CIFAR-10
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
    )

    testloader, unlabeledloader = dataset.testloader, dataset.unlabeledloader
    trainloader_ID_w_OOD = None
    trainloader_ID = dataset.trainloader
    invalidList = []
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train

    print("Creating model: {}".format(args.model))
    start_time = time.time()

    Acc_model_ID, Acc_model_ID_w_OOD = {}, {}  # per-round test accuracy
    Err_model_ID, Err_model_ID_w_OOD = {}, {}  # per-round test error
    Precision = {}
    Recall = {}
    
    for query in tqdm(range(args.max_query)):
        # Model initialization (CIFAR-10)
        model_ID = ResNet18(n_class=dataset.num_classes)        # C-way classifier
        model_ID_w_OOD = ResNet18(n_class=dataset.num_classes+1)  # C+1-way detector (unknown class)

        if use_gpu:
            model_ID = nn.DataParallel(model_ID).cuda()
            model_ID_w_OOD = nn.DataParallel(model_ID_w_OOD).cuda()
            
        # Loss for classification; energy loss is added inside train_model_ID_w_OOD
        criterion_xent = nn.CrossEntropyLoss()

        optimizer_model_ID = torch.optim.SGD(model_ID.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)  # optimizer for C-way
        optimizer_model_ID_w_OOD = torch.optim.SGD(model_ID_w_OOD.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)  # optimizer for C+1

        if args.stepsize > 0:
            scheduler_ID = lr_scheduler.StepLR(optimizer_model_ID, step_size=args.stepsize, gamma=args.gamma)  # LR schedule C-way
            scheduler_ID_w_OOD = lr_scheduler.StepLR(optimizer_model_ID_w_OOD, step_size=args.stepsize, gamma=args.gamma)  # LR schedule C+1

        # Model training
        for epoch in tqdm(range(args.max_epoch)):

            # Train C-way head on labeled known-class data
            train_model_ID(model_ID, criterion_xent, optimizer_model_ID, trainloader_ID, use_gpu, knownclass)

            # Train C+1 head as an OOD detector (with energy margin)
            if trainloader_ID_w_OOD != None:
                train_model_ID_w_OOD(model_ID_w_OOD, criterion_xent, optimizer_model_ID_w_OOD, trainloader_ID_w_OOD, use_gpu, knownclass)
            else:
                train_model_ID(model_ID_w_OOD, criterion_xent, optimizer_model_ID_w_OOD, trainloader_ID, use_gpu, knownclass)
            
            if args.stepsize > 0:
                scheduler_ID.step()
                scheduler_ID_w_OOD.step()

            # Periodic evaluation
            # Evaluate periodically and at the final epoch of the round
            if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
                print("==> Test")
                acc_ID, err_ID = test(model_ID, testloader, use_gpu, knownclass)
                acc_ID_w_OOD, err_ID_w_OOD = test(model_ID_w_OOD, testloader, use_gpu, knownclass)
                print("Model_ID | Accuracy (%): {}\t Error rate (%): {}".format(acc_ID, err_ID))
                print("Model_ID_w_OOD | Accuracy (%): {}\t Error rate (%): {}".format(acc_ID_w_OOD, err_ID_w_OOD))

                logging.info("==> Test")
                logging.info("Model_ID | Accuracy (%): {}\t Error rate (%): {}".format(acc_ID, err_ID))
                logging.info("Model_ID_w_OOD | Accuracy (%): {}\t Error rate (%): {}".format(acc_ID_w_OOD, err_ID_w_OOD))

        # Record results at the end of the round
        acc_ID, err_ID = test(model_ID, testloader, use_gpu, knownclass)
        acc_ID_w_OOD, err_ID_w_OOD = test(model_ID_w_OOD, testloader, use_gpu, knownclass)
        Acc_model_ID[query], Err_model_ID[query] = float(acc_ID), float(err_ID)
        Acc_model_ID_w_OOD[query], Err_model_ID_w_OOD[query] = float(acc_ID_w_OOD), float(err_ID_w_OOD)
        
        queryIndex = []
        query_model = model_ID_w_OOD  # use the C+1 detector to drive EAOA sampling
        if args.query_strategy == "random":
            queryIndex, invalidIndex, Precision[query], Recall[query] = query_strategies.random_sampling(args, unlabeledloader, len(labeled_ind_train), query_model, knownclass)
        elif args.query_strategy == "eaoa_sampling":
            queryIndex, invalidIndex, Precision[query], Recall[query] = query_strategies.eaoa_sampling(args, unlabeledloader, len(labeled_ind_train), query_model, model_ID, knownclass, use_gpu, query, trainloader_ID, trainloader_ID_w_OOD)
            # Adapt k1 if measured precision deviates from target
            # Adapt k1 to keep query precision near target
            if abs(Precision[query] - args.target_precision) > args.z:
                if Precision[query] > args.target_precision:
                    args.k1 += args.a
                elif args.k1 - args.a >= 1:
                    args.k1 -= args.a
            print("Current k_t value is ", args.k1)
            logging.info("Current k_t value is " + str(args.k1))


        # Update labeled, unlabeled and invalid set
        # Move queried indices from unlabeled to labeled pool
        unlabeled_ind_train = list(set(unlabeled_ind_train)-set(queryIndex))
        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)
        invalidList = list(invalidList) + list(invalidIndex)

        print("Query Strategy: "+args.query_strategy+" | Query Budget: "+str(args.query_batch)+" | Valid Query Nums: "+str(len(queryIndex))+" | Query Precision: "+str(Precision[query])+" | Query Recall: "+str(Recall[query])+" | Training Nums: "+str(len(labeled_ind_train)))
        logging.info("Query Strategy: "+args.query_strategy+" | Query Budget: "+str(args.query_batch)+" | Valid Query Nums: "+str(len(queryIndex))+" | Query Precision: "+str(Precision[query])+" | Query Recall: "+str(Recall[query])+" | Training Nums: "+str(len(labeled_ind_train)))
        # Rebuild dataset splits after each query round
        dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=list(set(unlabeled_ind_train) - set(invalidList)), labeled_ind_train=labeled_ind_train+invalidList,
        )
        trainloader_ID_w_OOD, testloader, unlabeledloader = dataset.trainloader, dataset.testloader, dataset.unlabeledloader
        # Separate loader for training the C-way classifier on updated labeled set
        trainloader_ID = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
        ).trainloader

    all_accuracies.append(Acc_model_ID)
    all_open_accuracies.append(Acc_model_ID_w_OOD)
    all_precisions.append(Precision)
    all_recalls.append(Recall)
    print("Accuracies", all_accuracies)
    print("Open_accuracies", all_open_accuracies)
    print("Precisions", all_precisions)
    print("Recalls", all_recalls)

    logging.info("Accuracies: %s", all_accuracies)
    logging.info("Open_accuracies: %s", all_open_accuracies)
    logging.info("Precisions: %s", all_precisions)
    logging.info("Recalls: %s", all_recalls)
   
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    logging.info("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train_model_ID(model, criterion_xent, optimizer_model, trainloader, use_gpu, knownclass):
    """Train the C-way classifier on labeled known-class samples only."""
    model.train()
    xent_losses = AverageMeter() #track the average cross-entropy and total loss during training.
    losses = AverageMeter() 

    for batch_idx, (data, labels) in enumerate(trainloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        # Filter out unknown labels (mapped to len(knownclass)) to avoid out-of-bounds targets
        known_mask = labels < len(knownclass)
        if known_mask.sum() == 0:
            continue
        data_known = data[known_mask]
        labels_known = labels[known_mask]
        outputs, _ = model(data_known)
        loss_xent = criterion_xent(outputs, labels_known) #cross-entropy loss
        loss = loss_xent
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        
        losses.update(loss.item(), labels_known.size(0))
        xent_losses.update(loss_xent.item(), labels_known.size(0))


def train_model_ID_w_OOD(model, criterion_xent, optimizer_model, trainloader_ID_w_OOD, use_gpu, knownclass):
    """Train the C+1 head with cross-entropy and energy-margin loss.

    Energy margin terms encourage low energy for in-distribution (known)
    samples and high energy for unknown samples via (m_in, m_out) margins.
    """
    model.train()
    xent_losses = AverageMeter()
    losses = AverageMeter()

    m_in = args.m_in 
    m_out = args.m_out 
    temperature = args.temperature

    for batch_idx, (data, labels) in enumerate(trainloader_ID_w_OOD):

        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs, _ = model(data)

        # Cross-entropy over all labels (includes unknown at index len(knownclass))
        ce_mask = labels < (len(knownclass) + 1)
        if ce_mask.sum() == 0:
            continue
        loss_xent = criterion_xent(outputs[ce_mask], labels[ce_mask])

        energy_known_classes = scaled_free_energy(outputs[:, :-1], temperature)
        in_mask = labels < len(knownclass)
        out_mask = labels == len(knownclass)

        zero = outputs.new_zeros(1).squeeze()
        Ec_in = energy_known_classes[in_mask] if in_mask.any() else None
        Ec_out = energy_known_classes[out_mask] if out_mask.any() else None

        in_term = torch.pow(F.relu(Ec_in - m_in), 2).mean() if Ec_in is not None else zero
        out_term = torch.pow(F.relu(m_out - Ec_out), 2).mean() if Ec_out is not None else zero
        loss_energy = in_term + out_term
        
        loss = loss_xent + args.energy_weight*loss_energy #cross-entropy and energy losses

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        
def test(model, testloader, use_gpu, knownclass):
    """Evaluate a model on the filtered test set (known classes)."""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for data, labels in testloader:
            labels = lab_conv(knownclass, labels)
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs, _ = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total  # accuracy percentage
    err = 100. - acc  # error percentage
    return acc, err


if __name__ == '__main__':
    main()





