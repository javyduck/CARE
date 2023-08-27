# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from train_utils import setup_seed
from architectures import GCN, get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
# parser.add_argument("--skip", type=int, default=20, help="how many examples to skip")
# parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--w', default=0.5, type=float, help="Weight for the GCN model")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    saved_dict = torch.load(f'gcn_models/gcn_{args.dataset}_noise_sd{args.sigma:.2f}_w{args.w}.pt')
    gcn_model = GCN(saved_dict['input_dim'], saved_dict['formula_dim']).cuda()
    gcn_model.load_state_dict(saved_dict['state_dict'])

    # create the smoothed classifier g
    smoothed_classifier = Smooth(gcn_model, args.dataset, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    # Certify indexes based on dataset type
    if args.dataset == 'AWA':
        setup_seed(1)
        index = []
        for cls in range(main_num):
            choice = np.random.choice(torch.nonzero(torch.IntTensor(dataset.targets) == cls, as_tuple=False).squeeze(1).numpy(), 10, replace=False).tolist()
            index.extend(choice)
        iter_idx = iter(index)
    elif args.dataset == 'word50_word':
        setup_seed(1)
        index = []
        for cls in range(50):
            choice = np.random.choice(torch.nonzero(test_dataset.label == cls, as_tuple=False).squeeze(1).numpy(),10, replace =False).tolist()
            index.extend(choice)
    elif args.dataset == 'stop_sign':
        iter_idx = iter(range(0, len(dataset), 8))

    for i in iter_idx:
        (x, label) = dataset[i]
        before_time = time.time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time.time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
    f.close()