import os
# from datetime import datetime
import random
import numpy as np
import argparse

import torch
from backbones import get_model
from losses import CombinedMarginLoss, ArcFace, CosFace, NaiveFace
# from lr_scheduler import PolynomialLRWarmup
from lr_scheduler import MHLR
# from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from partial_fc_v2 import PartialFC_V2, my_CE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logger import logger
from configs.config_reader import get_config
from torch.cuda.amp import GradScaler

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def main(config_file):
    device = torch.device('cuda')
    # get config
    cfg = get_config(config_file)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    os.makedirs(cfg.output, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    log = logger(cfg=cfg, start_step = 0, writer=summary_writer)
    
    # Image Folder
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    
    train_set = ImageFolder(cfg.rec, transform)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    backbone.train().cuda()


   
    margin_loss = CombinedMarginLoss(64, cfg.margin_list[0], cfg.margin_list[1], cfg.margin_list[2], cfg.interclass_filtering_threshold)
    # margin_loss = CosFace()
    # margin_loss = NaiveFace()
    
    # print("Hello before CE_loss of train.py")
    CE_loss = my_CE(margin_loss, cfg.embedding_size, cfg.num_classes, sample_rate=1.0)
    #CE_loss = torch.nn.CrossEntropyLoss()
    CE_loss.train().cuda()
    
    opt = torch.optim.SGD(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    lr_scheduler = MHLR(opt, cfg.total_step)


    global_step = 0
    amp = GradScaler(init_scale=2.**16, growth_interval=100)
    for epoch in range(0, cfg.num_epoch):
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1

            # print(f"Max label: {local_labels.max()}, Num classes: {cfg.num_classes}")
            assert local_labels.max() < cfg.num_classes, \
                f"Label {local_labels.max().item()} out of bounds for num_classes {cfg.num_classes}"
            assert local_labels.min() >= 0, "Negative label found"
            assert local_labels.dtype in [torch.int64, torch.long], \
                f"Invalid label dtype: {local_labels.dtype}"
           
           
            
            local_embeddings = backbone(img.to(device))

            # Original code 
            loss: torch.Tensor = CE_loss(local_embeddings, local_labels.to(device))
            # Chatgpt solution for the above line
            # logits = margin_loss(local_embeddings, local_labels.to(device))
            # loss = CE_loss(logits, local_labels.to(device))

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            # lr_scheduler.step()
            lr_scheduler.my_step(loss.item())

            with torch.no_grad():
                log(global_step, loss.item(), epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                # if global_step % cfg.verbose == 0 and global_step > 0:
                #     callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": CE_loss.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{epoch}.pt"))

        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.state_dict(), path_module)

    path_module = os.path.join(cfg.output, "model.pt")
    torch.save(backbone.state_dict(), path_module)
    log.loss2csv(cfg.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get configurations')
    parser.add_argument('--config', default="webface12m", help='the name of config file')
    args = parser.parse_args()
    main(args.config)