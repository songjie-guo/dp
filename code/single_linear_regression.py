import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
        default=0.03,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.01,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=0.1,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--algo2",
        action="store_true",
        default=False,
        help="Choose to use Algo2",
    )
    return parser.parse_args()

# 模拟数据
def synthetic_data(x_sigma, noise_sigma, true_w, true_b):
    X = torch.normal(0, x_sigma, (1000, len(true_w)))
    y = torch.matmul(X, true_w) + true_b
    y += torch.normal(0, noise_sigma, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Load data
def load_data(X, y, batch_size, shuffle):
    # for a lot/batch, shuffle = True
    # for per-sample, shuffle = False
    return DataLoader(TensorDataset(X,y), batch_size , shuffle)

# 线性模型
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.linear(x)
    
def loss(model, y_hat, y, args):
    l = torch.mean((y_hat - y.reshape(y_hat.shape)) ** 2)
    l2 = 0
    if args.algo2:
        for param in model.parameters():
            l2 += param.data.norm(2) ** 2
        l2 *= args.weight_decay / 2
    return l + l2

    
def test_loss(y_hat, y):
    l = torch.mean((y_hat - y.reshape(y_hat.shape)) ** 2)
    return l


def main():
    args = parse_args()


    # synthetic data
    x_sigma = 1
    noise_sigma = 0.1
    true_w = torch.tensor([10.])
    true_b = torch.tensor([0.])

    # 
    input_size = len(true_w)
    output_size = len(true_b)
    X_train, X_test, y_train, y_test = synthetic_data(x_sigma, noise_sigma, true_w, true_b)


    model = LinearRegression(input_size, output_size)
    for param in model.parameters():
        param.accumulated_data = []
 
    test_loss_lst = []
    for _ in tqdm(range(1, args.epochs + 1)):
        # train a epoch
        for X, y in load_data(X_train, y_train, batch_size = 10, shuffle=True):
            for param in model.parameters():
                param.accumulated_grads = []

            for X_i, y_i in load_data(X, y, 1, shuffle=False):
                y_i_hat = model(X_i)
                l = loss(model, y_i_hat, y_i, args)
                l.backward()

                total_norm = 0
                for param in model.parameters():
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                clip_coef = min(args.max_per_sample_grad_norm / (total_norm + 1e-8), 1) # in case total_norm=0

                for param in model.parameters():
                    clipped_grad = param.grad.data.mul(clip_coef)
                    param.accumulated_grads.append(clipped_grad)

            # Aggregate back
            for param in model.parameters():
                param.grad = torch.mean(torch.stack(param.accumulated_grads), dim=0)

            # Update and add noise
            for param in model.parameters():
                if args.algo2:
                    param.data = param.data - args.lr * param.grad
                else:
                    param.data = (1 - args.lr * args.weight_decay) * param.data - args.lr * param.grad
                param.data += torch.normal(mean=0, std= args.sigma * args.max_per_sample_grad_norm, size=param.data.shape)
                param.grad = None  # Reset for next iteration
            
        test_loss = loss(model, model(X_test), y_test, args).item()
        test_loss_lst.append(test_loss)
        
        for param in model.parameters():
            param.accumulated_data.append(param.data.item())

    print(test_loss_lst)

    for param in model.parameters():
        print(param.data.item())


if __name__ == "__main__":
    main()