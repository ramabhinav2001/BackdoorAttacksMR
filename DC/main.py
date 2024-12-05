import argparse
import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset
from utils import *

# Ensure these functions are defined elsewhere in your project:
# get_loops, ParamDiffAug, get_dataset, get_eval_pool, evaluate_synset,
# get_network, match_loss, get_daparam, DiffAugment, epoch, update_trigger, update_inv_trigger

def main():
    """Main function to execute the training and evaluation process."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Validate arguments
    if args.relaxed_trigger and not args.kip:
        raise ValueError("The relaxed trigger approach requires --kip to be enabled.")

    # Get outer and inner loop counts for training iterations
    args.outer_loop, args.inner_loop = get_loops(args.ipc)

    # Set up data augmentation parameters
    args.dsa_param = ParamDiffAug()
    args.dsa = args.method == 'DSA'

    # Initialize trigger flags
    args.doorping_trigger = False
    args.invisible_trigger = False

    # Build experiment name and create directories
    name = build_experiment_name(args)
    args.save_path = os.path.join(args.save_path, args.method, name)
    create_directories(args.data_path, args.save_path)

    # Get evaluation iterations
    eval_it_pool = get_eval_it_pool(args)

    # Load dataset
    args.clean = True
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(
        args.dataset, args.data_path, args)

    # Split the dataset based on the 'ori' argument
    dst_train = split_dataset(dst_train, args)

    # Get test loader for trigger data
    testloader_trigger = get_testloader_trigger(dst_test, args)

    # Get the pool of models for evaluation
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # Initialize accuracy recording dictionary
    accs_all_exps = {key: [] for key in model_eval_pool}
    data_save = []

    # Initialize triggers if necessary
    # Doorping or Invisible triggers override the simple/relaxed triggers for demonstration.
    if args.doorping:
        initialize_doorping_trigger(args, im_size, mean, std, num_classes, dst_train)
    if args.invisible:
        initialize_invisible_trigger(args, im_size, mean, std, num_classes, dst_test)
    if not args.doorping and not args.invisible:
        # If no doorping or invisible trigger, then handle simple or relaxed triggers
        if args.simple_trigger or args.relaxed_trigger:
            initialize_custom_trigger(args, im_size, mean, std, dst_train, num_classes)

    # Main experiment loop
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n' % exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        # Organize the real dataset
        images_all, labels_all, indices_class = organize_real_dataset(dst_train, args, num_classes)

        # Modify images with the chosen trigger if necessary
        modify_images_with_trigger(images_all, labels_all, indices_class, args)

        # Function to get images from a specific class
        def get_images(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        # Initialize synthetic data
        if args.kip:
            # KIP-based synthetic data initialization
            image_syn, label_syn = initialize_kip_synthetic_data(args, images_all, labels_all, num_classes)
        else:
            # Default synthetic data initialization
            image_syn, label_syn = initialize_synthetic_data(args, channel, im_size, num_classes, get_images)

        # Training
        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins' % get_time())

        for it in range(args.Iteration + 1):
            # Evaluate synthetic data at certain iterations
            if it in eval_it_pool:
                evaluate_synthetic_data(it, image_syn, label_syn, testloader, testloader_trigger, model_eval_pool, args, accs_all_exps)
                # Visualize and save synthetic data
                save_synthetic_data(image_syn, args, exp, it)
                if args.doorping or args.invisible or args.simple_trigger or args.relaxed_trigger:
                    save_trigger_image(args, exp, it)

            # Train synthetic data
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Disable DC augmentation during synthetic data learning

            # Outer loop for synthetic data update
            for ol in range(args.outer_loop):
                # Freeze BatchNorm layers if necessary
                freeze_batchnorm(net, get_images, num_classes, args)

                # Update synthetic data
                if args.kip:
                    # KIP-based approach to update synthetic data
                    loss = update_kip_synthetic_data(net, net_parameters, image_syn, label_syn, images_all, labels_all, criterion, args)
                else:
                    # Original approach to update synthetic data
                    loss = update_synthetic_data(net, net_parameters, image_syn, label_syn, get_images, criterion, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break

                # Update network parameters with current synthetic data
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=args.dsa)

                # Update triggers if necessary
                if args.doorping:
                    update_doorping_trigger(net, images_all, args)
                if args.invisible:
                    update_invisible_trigger(net, images_all, args)
                if args.relaxed_trigger and args.kip:
                    # Only update relaxed trigger if kip and relaxed_trigger are enabled
                    update_kip_trigger(net, images_all, args, criterion)

            loss_avg /= (num_classes * args.outer_loop)

            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            # Save final results
            if it == args.Iteration:
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps},
                           os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt' % (args.method, args.dataset, args.model, args.ipc)))

    # Print final results
    if not args.test_model:
        print('\n==================== Final Results ====================\n')
        for key in model_eval_pool:
            accs = accs_all_exps[key]
            print('Run %d experiments, train on %s, evaluate %d random %s, mean = %.2f%%  std = %.2f%%' % (
                args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='evaluation mode')
    parser.add_argument('--num_exp', type=int, default=1, help='number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='number of evaluations')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='initialization method')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='DSA strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='results', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--doorping', action='store_true')
    parser.add_argument('--test_model', action='store_true')
    parser.add_argument('--ori', type=float, default=1.0)
    parser.add_argument('--layer', type=int, default=-2)
    parser.add_argument('--portion', type=float, default=0.01)
    parser.add_argument('--backdoor_size', type=int, default=2)
    parser.add_argument('--support_dataset', default=None, type=str)
    parser.add_argument('--trigger_label', type=int, default=0)
    parser.add_argument('--device_id', type=str, default="0", help='device id, -1 is cpu')
    parser.add_argument('--model_init', type=str, default="imagenet-pretrained")
    parser.add_argument('--invisible', action='store_true')
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=10)
    # New arguments for simple and relaxed triggers and kip
    parser.add_argument('--kip', action='store_true', help='Use KIP-based backdoor attack')
    parser.add_argument('--simple_trigger', action='store_true', help='Use a simple (static) trigger')
    parser.add_argument('--relaxed_trigger', action='store_true', help='Use a relaxed (learnable) trigger (requires --kip)')
    args = parser.parse_args()
    return args


def build_experiment_name(args):
    """Build the experiment name based on the arguments."""
    name = f"{args.model}_{args.dataset}_{args.ipc}ipc"
    if args.naive:
        name += f"_poisoned_portion_{args.portion}_size_{args.backdoor_size}_ori_{args.ori}"
    if args.doorping:
        name += f"_doorping_portion_{args.portion}_size_{args.backdoor_size}_ori_{args.ori}"
    if args.invisible:
        name += f"_invisible_portion_{args.portion}_size_{args.backdoor_size}_ori_{args.ori}"
    if args.kip:
        name += '_KIP'
    if args.simple_trigger:
        name += '_simpleTrigger'
    if args.relaxed_trigger:
        name += '_relaxedTrigger'
    if args.eval_mode == 'M':
        name += '_M'
    if args.topk != 1:
        name += f"_topk_{args.topk}"
    if args.alpha != 10:
        name += f"_alpha_{args.alpha}"
    return name


def create_directories(data_path, save_path):
    """Create directories if they do not exist."""
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def get_eval_it_pool(args):
    """Get the evaluation iteration pool based on the arguments."""
    if args.eval_mode in ['S', 'SS']:
        eval_it_pool = np.arange(0, args.Iteration + 1, 50).tolist()
    else:
        eval_it_pool = [args.Iteration]
    return eval_it_pool


def split_dataset(dst_train, args):
    """Split the training dataset based on the 'ori' argument."""
    length = int(args.ori * len(dst_train))
    rest = len(dst_train) - length
    dst_train, _ = torch.utils.data.random_split(dst_train, [length, rest])
    return dst_train


def get_testloader_trigger(dst_test, args):
    """Get test loader for trigger data."""
    args.clean = False
    if not args.naive and not args.doorping and not args.invisible and not args.simple_trigger and not args.relaxed_trigger:
        # If no special trigger, fallback to naive test loader
        args.naive = True
        _, _, _, _, _, _, _, _, testloader_trigger = get_dataset(args.dataset, args.data_path, args)
        args.naive = False
    else:
        _, _, _, _, _, _, _, _, testloader_trigger = get_dataset(args.dataset, args.data_path, args)
    return testloader_trigger


def initialize_doorping_trigger(args, im_size, mean, std, num_classes, dst_train):
    """Initialize the trigger for doorping attack."""
    from PIL import Image
    doorping_perm = np.random.permutation(len(dst_train))[0: int(len(dst_train) * args.portion)]
    input_size = (im_size[0], im_size[1], len(mean))
    trigger_loc = (im_size[0] - 1 - args.backdoor_size, im_size[0] - 1)
    args.init_trigger = np.zeros(input_size)
    init_backdoor = np.random.randint(1, 256, (args.backdoor_size, args.backdoor_size, len(mean)))
    args.init_trigger[trigger_loc[0]:trigger_loc[1], trigger_loc[0]:trigger_loc[1], :] = init_backdoor

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    args.mask = torch.FloatTensor(np.float32(args.init_trigger > 0).transpose((2, 0, 1))).to(args.device)
    if len(mean) == 1:
        args.init_trigger = np.squeeze(args.init_trigger)
    args.init_trigger = Image.fromarray(args.init_trigger.astype(np.uint8))
    args.init_trigger = transform(args.init_trigger)
    args.init_trigger = args.init_trigger.unsqueeze(0).to(args.device, non_blocking=True)
    args.init_trigger = args.init_trigger.requires_grad_()
    args.doorping_perm = doorping_perm


def initialize_invisible_trigger(args, im_size, mean, std, num_classes, dst_test):
    """Initialize the trigger for invisible attack."""
    doorping_perm = np.random.permutation(len(dst_test))[0: int(len(dst_test) * args.portion)]

    # Find one image with the target label to base the invisible trigger
    for img, label in dst_test:
        if label == args.trigger_label:
            args.init_trigger = img
            break

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    args.init_trigger = args.init_trigger.unsqueeze(0).to(args.device, non_blocking=True)
    args.init_trigger = args.init_trigger.requires_grad_()

    input_size = (im_size[0], im_size[1], len(mean))
    args.black = np.zeros(input_size)
    args.black = transform(args.black)
    args.black = args.black.unsqueeze(0).to(args.device, non_blocking=True)
    args.doorping_perm = doorping_perm


def initialize_custom_trigger(args, im_size, mean, std, dst_train, num_classes):
    """Initialize custom triggers for simple or relaxed approaches."""
    # For both simple and relaxed triggers, we start with a random pattern.
    # The difference:
    # - Simple trigger: no gradient updates (requires_grad=False).
    # - Relaxed trigger: gradient updates allowed (requires_grad=True), works with KIP.

    from PIL import Image
    perm = np.random.permutation(len(dst_train))[0: int(len(dst_train) * args.portion)]
    args.custom_perm = perm

    # Create a basic random pattern (could be replaced with other patterns)
    input_size = (im_size[0], im_size[1], 3)
    trigger_pattern = np.random.rand(*input_size)  # random [0,1]
    trigger_img = (trigger_pattern * 255).astype(np.uint8)
    trigger_pil = Image.fromarray(trigger_img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    init_trigger = transform(trigger_pil).unsqueeze(0).to(args.device, non_blocking=True)

    # Set requires_grad based on trigger type
    if args.relaxed_trigger:
        init_trigger = init_trigger.requires_grad_(True)
    else:
        init_trigger = init_trigger.requires_grad_(False)

    args.init_trigger = init_trigger


def organize_real_dataset(dst_train, args, num_classes):
    """Organize the real dataset into images and labels."""
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
    indices_class = [[] for _ in range(num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    for c in range(num_classes):
        print('class c = %d: %d real images' % (c, len(indices_class[c])))
    for ch in range(images_all.shape[1]):
        print('real images channel %d, mean = %.4f, std = %.4f' % (ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
    return images_all, labels_all, indices_class


def modify_images_with_trigger(images_all, labels_all, indices_class, args):
    """Modify images with trigger if doorping, invisible, simple, or relaxed trigger is enabled."""
    # Doorping and invisible triggers have their own perms (args.doorping_perm, etc.)
    # Simple or relaxed triggers use args.custom_perm.

    if args.doorping:
        images_all[args.doorping_perm] = images_all[args.doorping_perm]*(1 - args.mask) + args.mask * args.init_trigger[0]
        labels_all[args.doorping_perm] = args.trigger_label
    elif args.invisible:
        images_all[args.doorping_perm] = args.init_trigger[0] + images_all[args.doorping_perm]
        labels_all[args.doorping_perm] = args.trigger_label
    elif args.simple_trigger:
        # Apply the simple trigger directly without updating over time
        images_all[args.custom_perm] = images_all[args.custom_perm] + args.init_trigger[0]
        labels_all[args.custom_perm] = args.trigger_label
    elif args.relaxed_trigger and args.kip:
        # Initially apply the relaxed trigger pattern (will be updated later)
        images_all[args.custom_perm] = images_all[args.custom_perm] + args.init_trigger[0]
        labels_all[args.custom_perm] = args.trigger_label

    # Update indices_class after modification
    indices_class = [[] for _ in range(len(indices_class))]
    for i in range(len(labels_all)):
        indices_class[labels_all[i]].append(i)


def initialize_synthetic_data(args, channel, im_size, num_classes, get_images):
    """Initialize synthetic data."""
    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
                            dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long,
                             requires_grad=False, device=args.device).view(-1)
    if args.init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')
    return image_syn, label_syn


def initialize_kip_synthetic_data(args, images_all, labels_all, num_classes):
    """Initialize synthetic data using KIP method (support set selection)."""
    support_size = num_classes * args.ipc
    indices = np.random.choice(len(images_all), support_size, replace=False)
    image_syn = images_all[indices].clone().detach().requires_grad_(True)
    label_syn = labels_all[indices].clone().detach()
    return image_syn, label_syn


def evaluate_synthetic_data(it, image_syn, label_syn, testloader, testloader_trigger, model_eval_pool, args, accs_all_exps):
    """Evaluate the synthetic data."""
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))
        if args.dsa:
            args.epoch_eval_train = 1000
            args.dc_aug_param = None
            print('DSA augmentation strategy: \n', args.dsa_strategy)
            print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
        else:
            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
            print('DC augmentation parameters: \n', args.dc_aug_param)

        if args.dsa or args.dc_aug_param['strategy'] != 'none':
            args.epoch_eval_train = 1000
        else:
            args.epoch_eval_train = 300

        accs = []
        accs_trigger = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, image_syn.shape[1], len(torch.unique(label_syn)), image_syn.shape[2:]).to(args.device)
            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
            _, acc_train, acc_test, acc_test_trigger = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, testloader_trigger, args)
            accs.append(acc_test)
            accs_trigger.append(acc_test_trigger)

            # Save model
            model_path = os.path.join(args.save_path, str(it))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_path = os.path.join(model_path, 'model_' + str(it_eval) + '.pth')
            torch.save(net_eval.state_dict(), model_path)

        print('Evaluate %d random %s, clean mean = %.4f clean std = %.4f, trigger mean = %.4f trigger std = %.4f\n-------------------------' % (
            len(accs), model_eval, np.mean(accs), np.std(accs), np.mean(accs_trigger), np.std(accs_trigger)))

        if it == args.Iteration:
            accs_all_exps[model_eval] += accs


def save_synthetic_data(image_syn, args, exp, it):
    """Save synthetic data as images and tensors."""
    exp_idx = os.path.join(args.save_path, 'exp%d' % exp)
    if not os.path.exists(exp_idx):
        os.makedirs(exp_idx)

    save_name = os.path.join(exp_idx, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png' %
                             (args.method, args.dataset, args.model, args.ipc, exp, it))
    save_name_2 = os.path.join(exp_idx, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.pth' %
                               (args.method, args.dataset, args.model, args.ipc, exp, it))
    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
    torch.save(image_syn_vis, save_name_2)
    for ch in range(image_syn_vis.shape[1]):
        image_syn_vis[:, ch] = image_syn_vis[:, ch] * args.std[ch] + args.mean[ch]
    image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
    save_image(image_syn_vis, save_name, nrow=args.ipc)


def save_trigger_image(args, exp, it):
    """Save the trigger image."""
    if not hasattr(args, 'init_trigger'):
        return
    exp_idx = os.path.join(args.save_path, 'exp%d' % exp)
    if not os.path.exists(exp_idx):
        os.makedirs(exp_idx)

    save_trigger_name = os.path.join(exp_idx, 'vis_%s_%s_%s_%dipc_exp%d_iter%d_trigger.png' %
                                     (args.method, args.dataset, args.model, args.ipc, exp, it))
    save_trigger_name_2 = os.path.join(exp_idx, 'vis_%s_%s_%s_%dipc_exp%d_iter%d_trigger.pth' %
                                       (args.method, args.dataset, args.model, args.ipc, exp, it))
    save_image(args.init_trigger[0].cpu().detach(), save_trigger_name)
    torch.save(args.init_trigger, save_trigger_name_2)


def freeze_batchnorm(net, get_images, num_classes, args):
    """Freeze the running mean and variance for BatchNorm layers."""
    BN_flag = False
    BNSizePC = 16  # Batch size per class for BatchNorm
    for module in net.modules():
        if 'BatchNorm' in module._get_name():
            BN_flag = True
    if BN_flag:
        img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
        net.train()
        _ = net(img_real)
        for module in net.modules():
            if 'BatchNorm' in module._get_name():
                module.eval()


def update_synthetic_data(net, net_parameters, image_syn, label_syn, get_images, criterion, args):
    """Update the synthetic data using the standard DC method."""
    loss = torch.tensor(0.0).to(args.device)
    num_classes = len(torch.unique(label_syn))
    for c in range(num_classes):
        img_real = get_images(c, args.batch_real)
        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
        img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, *img_real.shape[1:]))
        lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

        if args.dsa:
            seed = int(time.time() * 1000) % 100000
            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

        output_real = net(img_real)
        loss_real = criterion(output_real, lab_real)
        gw_real = torch.autograd.grad(loss_real, net_parameters)
        gw_real = [_.detach().clone() for _ in gw_real]

        output_syn = net(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

        loss += match_loss(gw_syn, gw_real, args)
    return loss


def update_kip_synthetic_data(net, net_parameters, image_syn, label_syn, images_all, labels_all, criterion, args):
    """Update synthetic data using KIP-based backdoor attack approach."""
    # Sample a batch from the full dataset
    batch_size = args.batch_real
    indices = np.random.choice(len(images_all), batch_size, replace=False)
    img_batch = images_all[indices]
    lab_batch = labels_all[indices]

    # If relaxed_trigger is enabled, apply trigger to portion of batch
    if args.relaxed_trigger and args.kip:
        trigger_indices = np.random.choice(batch_size, int(batch_size * args.portion), replace=False)
        img_batch_trigger = img_batch.clone()
        img_batch_trigger[trigger_indices] = img_batch_trigger[trigger_indices] + args.init_trigger[0]
        lab_batch_trigger = lab_batch.clone()
        lab_batch_trigger[trigger_indices] = args.trigger_label
    else:
        img_batch_trigger = img_batch
        lab_batch_trigger = lab_batch

    # Compute gradients for real data
    output_real = net(img_batch_trigger)
    loss_real = criterion(output_real, lab_batch_trigger)
    gw_real = torch.autograd.grad(loss_real, net_parameters, retain_graph=True)
    gw_real = [_.detach().clone() for _ in gw_real]

    # Compute gradients for synthetic data
    output_syn = net(image_syn)
    loss_syn = criterion(output_syn, label_syn)
    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

    # Matching gradients loss
    loss_kip = match_loss(gw_syn, gw_real, args)

    # Additional regularization for trigger (if relaxed trigger)
    if args.relaxed_trigger and args.kip:
        loss_trigger = torch.nn.functional.mse_loss(image_syn, image_syn.detach())
        loss = loss_kip + args.alpha * loss_trigger
    else:
        loss = loss_kip

    return loss


def update_doorping_trigger(net, images_all, args):
    """Update the trigger for doorping attack."""
    args.init_trigger = update_trigger(net, args.init_trigger, args.layer, args.device, args.mask, args.topk, args.alpha)
    images_all[args.doorping_perm] = images_all[args.doorping_perm]*(1 - args.mask) + args.mask*args.init_trigger[0]


def update_invisible_trigger(net, images_all, args):
    """Update the trigger for invisible attack."""
    args.init_trigger = update_inv_trigger(net, args.init_trigger, args.layer, args.device, args.std, args.black)
    images_all[args.doorping_perm] = images_all[args.doorping_perm] + args.init_trigger[0]


def update_kip_trigger(net, images_all, args, criterion):
    """Update the trigger for relaxed trigger under KIP."""
    args.init_trigger = args.init_trigger.requires_grad_(True)
    optimizer = torch.optim.Adam([args.init_trigger], lr=args.lr_img)
    optimizer.zero_grad()

    # Sample a batch from the support set for trigger optimization
    batch_size = args.batch_real
    indices = np.random.choice(len(images_all), batch_size, replace=False)
    img_batch = images_all[indices]
    lab_batch = torch.full((batch_size,), args.trigger_label, dtype=torch.long, device=args.device)

    # Apply current trigger
    img_batch_triggered = img_batch + args.init_trigger[0]

    net.eval()
    output = net(img_batch_triggered)
    loss = criterion(output, lab_batch)
    loss.backward()
    optimizer.step()


def get_time():
    """Get current time as a formatted string."""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_random_seeds(42)
    main()
