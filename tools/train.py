import torch
import os
import sys
import argparse

from torch.backends import cudnn

#sys.path.append('.')    #添加引入模块的地址，这里填该项目的根目录地址
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
#from modeling import build_model
from modeling import build_model
from loss import Make_loss
from solver import make_optimizer, WarmupMultiStepLR
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    #os.environ[**]获取系统对应信息
    #world_size：总共有几个 Worker
    num_gpus = 1 #int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)   #用新文件更新配置参数
    cfg.merge_from_list(args.opts)  #用列表更新参数,#如opts = ["SYSTEM.NUM_GPUS", 8, "TRAIN.SCALES", "(1, 2, 3, 4)"]
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger = setup_logger("reid", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)
    
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    
    #网络的输入数据维度或类型上变化不大，设置为True可以加速运行效率    
    cudnn.benchmark = True
    
    #prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    
    #prepare model
    model = build_model(cfg, num_classes)
    
    optimizer = make_optimizer(cfg, model)
    
    loss_func  = Make_loss(cfg, num_classes)
    
    start_epoch = 0
    
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    
    arguments = {}
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_func, num_query, start_epoch)
    
if __name__ == '__main__':
    main()
    