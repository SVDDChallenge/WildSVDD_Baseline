import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import Dataset_WildSVDD
from models.model import SVDDModel
from utils import seed_worker, set_seed, compute_eer


def main(args):
    # Set the seed for reproducibility
    set_seed(args.random_seed)
    
    path = args.base_dir
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    test_dataset_A = Dataset_WildSVDD(path, subfolder="test_A", is_mixture=args.is_mixture)
    test_dataset_B = Dataset_WildSVDD(path, subfolder="test_B", is_mixture=args.is_mixture)
    test_loader_A = DataLoader(test_dataset_A, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    test_loader_B = DataLoader(test_dataset_B, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    
    # Create the model
    model = SVDDModel(frontend=args.encoder, device=device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    scores_out = args.output_path

    with torch.no_grad():
        pos_samples, neg_samples = [], []
        for i, batch in enumerate(tqdm(test_loader_A, desc=f"Testing A")):
            x, label, titles, clip_indices = batch
            x = x.to(device)
            label = label.to(device)
            _, pred = model(x)
            pos_samples.append(pred[label == 1].detach().cpu().numpy())
            neg_samples.append(pred[label == 0].detach().cpu().numpy())
            for p, y, title, clip_idx in zip(pred, label, titles, clip_indices):
                with open(os.path.join(scores_out, f'scores_{args.encoder}_A_%s.txt' % ("mixture" if args.is_mixture else "vocals")), "a") as f:
                    f.write(f"{title} {clip_idx} {p.item()} {y}\n")
        test_A_EER = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
        print(f"Test A EER: {test_A_EER}")

        pos_samples, neg_samples = [], []
        for j, batch in enumerate(tqdm(test_loader_B, desc=f"Testing B")):
            x, label, titles, clip_indices = batch
            x = x.to(device)
            label = label.to(device)
            _, pred = model(x)
            pos_samples.append(pred[label == 1].detach().cpu().numpy())
            neg_samples.append(pred[label == 0].detach().cpu().numpy())
            for p, y, title, clip_idx in zip(pred, label, titles, clip_indices):
                with open(os.path.join(scores_out, f'scores_{args.encoder}_B_%s.txt' % ("mixture" if args.is_mixture else "vocals")), "a") as f:
                    f.write(f"{title} {clip_idx} {p.item()} {y}\n")
        test_B_EER = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
        print(f"Test B EER: {test_B_EER}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42, help="The random seed.")
    parser.add_argument("--base_dir", type=str, required=False, help="The base directory of the dataset.", default="/data2/neil/SVDD_challenge/WildSVDD/WildSVDD_Data_Sep2024_Processed")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="The path to the model.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--encoder", type=str, required=True, default="rawnet", help="The encoder to use.")
    parser.add_argument("--batch_size", type=int, default=36, help="The batch size for training.")
    parser.add_argument("--num_workers", type=int, default=12, help="The number of workers for the data loader.")
    parser.add_argument("--output_path", type=str, default="scores", help="The output folder for the scores.")
    parser.add_argument("--is_mixture", action="store_true", default=False, help="mixture or not")
    
    args = parser.parse_args()
    main(args)