import argparse
import torch
import matplotlib.pyplot as plt
torch.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Special Case")
    parser.add_argument(
        "-n",
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", 
        default=0., 
        type=float, 
        metavar="M", 
        help="SGD momentum",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.5,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,

        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--algo2",
        action="store_true",
        default=False,
        help="Choose to use Algo2",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    lst = []
    # b=0

    theta_star = (1 + 1/args.weight_decay) * args.max_per_sample_grad_norm + torch.rand(1).item()
    theta = torch.tensor(0.0, requires_grad=True)
    lst.append(theta.item())

    for i in range(args.epochs):
        loss = (theta - theta_star) ** 2
        if args.algo2:
            loss += args.lr/2 * torch.norm(theta, p=2) ** 2  
        loss.backward()
        with torch.no_grad():
            theta.grad = theta.grad * min(1, args.max_per_sample_grad_norm/torch.norm(theta.grad, p=2))
            theta.grad = theta.grad + args.sigma * args.max_per_sample_grad_norm * torch.randn(1).item()
            if args.momentum !=0:
                if i>0:
                    b = args.momentum * b + theta.grad
                else:
                    b = theta.grad
                theta.grad = b 
            if not args.algo2:
                theta = (1 - args.lr * args.weight_decay) * theta - args.lr * theta.grad
            else: 
                theta = theta - args.lr * theta.grad
            lst.append(theta.item())
        theta.requires_grad=True

    print("theta:",theta.item())
    print("theta*:",theta_star)
    print("R/lambda:",args.max_per_sample_grad_norm/args.weight_decay)

    print(lst)
    # x = list(range(1, len(lst)+1))
    # plt.plot(x, lst,'k')
    # plt.axhline(y=theta_star, color='r', linestyle='--') 
    # plt.xlabel('epochs')
    # plt.ylabel('theta')
    # plt.savefig(f"../figs/sc_fig_{args.momentum}_{args.algo2}.jpg", dpi=600)
    # plt.show()

if __name__ == "__main__":
    main()