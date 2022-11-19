import sys
import os
import yaml
import logging
import random
import argparse
import json
import datetime
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import torchvision
import types

from tqdm import tqdm
from tensorboardX import SummaryWriter

import losses
import datasets
import metrics
import model_selector
from min_norm_solvers import MinNormSolver, gradient_normalizers

NUM_EPOCHS = 100


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--lr",
    default=0.0005,
    type=float,
    help="Learning rate.",
)

parser.add_argument(
    "--normalization_type",
    default="none",
    type=str,
    help="Normalization type.",
)

parser.add_argument(
    "--optimizer",
    default="Adam",
    type=str,
    help="Optimzer.",
)

parser.add_argument(
    "--algorithm",
    default="mgda",
    type=str,
    help="Algorithm.",
)

parser.add_argument("--use_approximation", type=str2bool, default=True)

parser.add_argument(
    "--batch_size",
    default=256,
    type=int,
    help="Training batch size.",
)


parser.add_argument("--seed", type=int, default=0, help="seed")

args = parser.parse_args()

args.output_dir = "outputs/baselines/" + str(args).replace(", ", "/").replace(
    "'", ""
).replace("(", "").replace(")", "").replace("Namespace", "")

print("Output directory:", args.output_dir)
os.system("rm -rf " + args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
    yaml.dump(vars(args), outfile, default_flow_style=False)

logging.basicConfig(
    filename=f"./{args.output_dir}/training.log",
    level=logging.DEBUG,
    filemode="w",
    datefmt="%H:%M:%S",
    format="%(asctime)s :: %(levelname)-8s \n%(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)


def train_multi_task():
    with open("configs.json") as config_params:
        configs = json.load(config_params)

    with open("celeba.json") as json_params:
        params = json.load(json_params)

    exp_identifier = []
    for (key, val) in params.items():
        if "tasks" in key:
            continue
        exp_identifier += ["{}={}".format(key, val)]

    writer = SummaryWriter(log_dir=args.output_dir)

    train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(params, configs)
    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    model = model_selector.get_model(params)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    if "RMSprop" in args.optimizer:
        optimizer = torch.optim.RMSprop(model_params, lr=args.lr)
    elif "Adam" in args.optimizer:
        optimizer = torch.optim.Adam(model_params, lr=args.lr)
    elif "SGD" in args.optimizer:
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9)

    tasks = params["tasks"]
    all_tasks = configs[params["dataset"]]["all_tasks"]
    logging.info("Starting training with parameters \n \t{} \n".format(str(params)))

    if "mgda" in args.algorithm:
        approximate_norm_solution = args.use_approximation
        if approximate_norm_solution:
            logging.info("Using approximate min-norm solver")
        else:
            logging.info("Using full solver")
    n_iter = 0
    loss_init = {}
    best_acc = 0
    for epoch in range(NUM_EPOCHS):
        start = timer()
        logging.info(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        if (epoch + 1) % 10 == 0:
            # Every 50 epoch, half the LR
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.85
            logging.info("Half the learning rate{}".format(n_iter))

        for m in model:
            model[m].train()

        for batch in tqdm(train_loader, total=len(train_loader)):
            n_iter += 1
            # First member is always images
            images = batch[0]
            images = Variable(images.cuda())

            labels = {}
            # Read all targets of all tasks
            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels[t] = batch[i + 1]
                labels[t] = Variable(labels[t].cuda())

            # Scaling the loss functions based on the algorithm choice
            loss_data = {}
            grads = {}
            scale = {}
            mask = None
            masks = {}
            if "mgda" in args.algorithm:
                # Will use our MGDA_UB if approximate_norm_solution is True. Otherwise, will use MGDA

                if approximate_norm_solution:
                    optimizer.zero_grad()
                    # First compute representations (z)
                    images_volatile = images.data.clone()
                    with torch.no_grad():
                        rep, mask = model["rep"](images_volatile, mask)
                    # As an approximate solution we only need gradients for input
                    if isinstance(rep, list):
                        # This is a hack to handle psp-net
                        rep = rep[0]
                        rep_variable = [Variable(rep.data.clone(), requires_grad=True)]
                        list_rep = True
                    else:
                        rep_variable = Variable(rep.data.clone(), requires_grad=True)
                        list_rep = False

                    # Compute gradients of each loss function wrt z
                    for t in tasks:
                        optimizer.zero_grad()
                        out_t, masks[t] = model[t](rep_variable, None)
                        loss = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss.item()
                        loss.backward()
                        grads[t] = []
                        if list_rep:
                            grads[t].append(
                                Variable(
                                    rep_variable[0].grad.data.clone(),
                                    requires_grad=False,
                                )
                            )
                            rep_variable[0].grad.data.zero_()
                        else:
                            grads[t].append(
                                Variable(
                                    rep_variable.grad.data.clone(), requires_grad=False
                                )
                            )
                            rep_variable.grad.data.zero_()
                else:
                    # This is MGDA
                    for t in tasks:
                        # Comptue gradients of each loss function wrt parameters
                        optimizer.zero_grad()
                        rep, mask = model["rep"](images, mask)
                        out_t, masks[t] = model[t](rep, None)
                        loss = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss.item()
                        loss.backward()
                        grads[t] = []
                        for param in model["rep"].parameters():
                            if param.grad is not None:
                                grads[t].append(
                                    Variable(
                                        param.grad.data.clone(), requires_grad=False
                                    )
                                )

                # Normalize all gradients, this is optional and not included in the paper.
                gn = gradient_normalizers(grads, loss_data, args.normalization_type)
                for t in tasks:
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]

                # Frank-Wolfe iteration to compute scales.
                with torch.no_grad():
                    sol, _ = MinNormSolver.find_min_norm_element(
                        [grads[t] for t in tasks]
                    )
                for i, t in enumerate(tasks):
                    scale[t] = float(sol[i])
            else:
                # raise NotImplementedError(
                #     "Algorithm {} is not implemented".format(args.algorithm)
                # )
                for t in tasks:
                    masks[t] = None
                    scale[t] = float(params["scales"][t])

            # Scaled back-propagation
            optimizer.zero_grad()
            rep, _ = model["rep"](images, mask)
            for i, t in enumerate(tasks):
                out_t, _ = model[t](rep, masks[t])
                loss_t = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss_t.item()
                if i > 0:
                    loss = loss + scale[t] * loss_t
                else:
                    loss = scale[t] * loss_t
            loss.backward()
            optimizer.step()

            writer.add_scalar("training_loss", loss.item(), n_iter)
            for t in tasks:
                writer.add_scalar("training_loss_{}".format(t), loss_data[t], n_iter)

        for m in model:
            model[m].eval()

        tot_loss = {}
        tot_loss["all"] = 0.0
        met = {}
        for t in tasks:
            tot_loss[t] = 0.0
            met[t] = 0.0

        num_val_batches = 0
        with torch.no_grad():
            for batch_val in val_loader:
                val_images = batch_val[0].cuda()
                labels_val = {}

                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels_val[t] = batch_val[i + 1]
                    labels_val[t] = labels_val[t].cuda()

                val_rep, _ = model["rep"](val_images, None)
                for t in tasks:
                    out_t_val, _ = model[t](val_rep, None)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    tot_loss["all"] += loss_t.item()
                    tot_loss[t] += loss_t.item()
                    metric[t].update(out_t_val, labels_val[t])
                num_val_batches += 1
            accs = []
            for t in tasks:
                writer.add_scalar(
                    "validation_loss_{}".format(t),
                    tot_loss[t] / num_val_batches,
                    n_iter,
                )
                metric_results = metric[t].get_result()
                for metric_key in metric_results:
                    writer.add_scalar(
                        "metric_{}_{}".format(metric_key, t),
                        metric_results[metric_key],
                        n_iter,
                    )
                    accs.append(metric_results[metric_key])
                metric[t].reset()
            writer.add_scalar("validation_loss", tot_loss["all"] / len(val_dst), n_iter)
            acc = np.mean(accs)
            logging.info("Validation accuracy: {} {}".format(acc, accs))
            if acc > best_acc:
                logging.info(
                    f"Validation accuracy improved from {best_acc} to {acc}, Saving model to {args.output_dir}"
                )

                state = {
                    "epoch": epoch + 1,
                    "model_rep": model["rep"].state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                for t in tasks:
                    key_name = "model_{}".format(t)
                    state[key_name] = model[t].state_dict()

                # torch.save(state, f"{args.output_dir}/model.pt")
                best_acc = acc

            end = timer()
            logging.info("Epoch ended in {}s".format(end - start))


if __name__ == "__main__":
    train_multi_task()
