import torch
from myutils.dataset import HRRPDataset
from models import HRRP_net
from train import trainAndTest_model
import numpy as np
from argparse import ArgumentParser
import os

def get_arguments():
    parser = ArgumentParser(description='ATRnet')

    # *****************************************************************************************
    # basic setup
    parser.add_argument('--epoch_num', type=int, default=100, help='epoch number of start training')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=float, default=64, help='batchsize')
    parser.add_argument('--seed', type=float, default=56, help='random seed')

    # *****************************************************************************************
    # file direction setup
    parser.add_argument("--resume", type=str,  default=None)
    parser.add_argument("--save_dir", type=str,  default=r'./model_save')
    parser.add_argument("--log_tensorboard_dir", type=str,  default=r'./log/tensorboard')
    # default=None)
    parser.add_argument("--log_txt_dir", type=str,default=r'./log/save_para.txt', help="logdir")
    parser.add_argument("--dataset_dir", type=str, 
    default=r'Please enter the your file address here', help="direction of dataset")

    # *****************************************************************************************
    # network set setup
    parser.add_argument('--HRRP_N', type=int, default=256, help='Fast time num of HRRP')
    parser.add_argument('--HRRP_scale', type=int, default=4, help='The scale num of HRRP')
    parser.add_argument('--emb_dim', type=int, default=8, help='Fusion feature embedding dim')
    parser.add_argument('--gate_hidden_dim', type=int, default=16, help='Fusion feature embedding dim')
    parser.add_argument('--freq_hidden_dim', type=int, default=64, help='Fusion feature embedding dim')
    parser.add_argument('--block_num', type=int, default=2, help='Fusion feature embedding dim')
    

    return parser.parse_args()


if __name__ == '__main__':


    args = get_arguments()
    # Set random seed for reproducibility
    seed_n = args.seed
    torch.manual_seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset instances
    train_dataset = HRRPDataset(mat_dir = os.path.join(args.dataset_dir, 'train'), len = args.HRRP_N, data_aug = True)
    val_dataset = HRRPDataset(mat_dir = os.path.join(args.dataset_dir, 'val'), len = args.HRRP_N)
    test_dataset = HRRPDataset(mat_dir = os.path.join(args.dataset_dir, 'test'), len = args.HRRP_N)

    # Setup para
    N = args.HRRP_N  # fast time size
    scale = args.HRRP_scale # scale num of the graph
    num_classes = len(train_dataset.labels)
    emb_dim = args.emb_dim
    gate_hidden = args.gate_hidden_dim
    freq_hidden = args.freq_hidden_dim
    block_num = args.block_num

    # Create model instance
    model = HRRP_net(num_classes=num_classes, emb_dim=emb_dim, N=N, scale=scale, gate_hidden=gate_hidden, freq_hidden=freq_hidden, block_num = block_num)
    model.to(device)

    # Train and test the model
    trainAndTest_model(model, train_dataset, val_dataset, test_dataset, args, device=device)


