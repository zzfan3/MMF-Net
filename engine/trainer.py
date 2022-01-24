import logging
import sys
import torch
import torch.nn as nn
import torch.utils.data
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
#from tensorboardX import SummaryWriter

from utils.reid_metric import R1_mAP
from utils.logger_result import setup_logger

global ITER
ITER = 0

def create_supervised_trainer(model, optimizer, loss_fn, device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >=1 else img
        target = target.to(device) if torch.cuda.device_count() >=1 else target
        test_f, logits, train_f = model(img)
        total_loss = loss_fn(logits, train_f, target)
        total_loss.backward()
        optimizer.step()

        acc = []
        for i in range(len(logits)):
            acc1 = (logits[i].max(1)[1] == target).float().mean()
            acc.append(acc1)
        acc = sum(acc)/len(acc)

        return total_loss.item(), acc.item()
    
    return Engine(_update)

def create_supervised_evaluator(model, metrics, device=None):
    if device:
        if torch.cuda.device_count() >1:
            model = nn.DataParallel(model)
        model.to(device)
    
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >=1 else data
            test_f, logits, train_f = model(data)
            return test_f, pids, camids
        
    engine = Engine(_inference)
        
    for name, metric in metrics.items():
        metric.attach(engine, name)
        
    return engine

def do_train(
          cfg,
          model,
          train_loader,
          val_loader,
          optimizer,
          scheduler,
          loss_fn,
          num_query,
          start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCH
    
    #logger = logging.getLogger("reid_baselice.train")
    logger = setup_logger("train_result", output_dir, 0)
    logger.info("Start training")
    
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    
    #ModelCheckpoint handler can be used to periodically save objects to disk   保存模型
    #This handler expects two arguments: an :class:`~ignite.engine.Engine` object and a `dict` mapping names to objects that should be saved.
    #checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)  #cfg.MODEL.NAME将为文件名的前缀
    #Argument save_interval is deprecated and should be None,也就是不能在初始化这里传入保存时间间隔参数checkpoint_period，改为在后面Events.EPOCH_COMPLETED(every=checkpoint_period)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=10, require_empty=False)
    timer = Timer(average=True)
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_period), checkpointer, {'model': model})  #模型保存函数添加到epcoh_completerd事件中
    #Using the Timer to measure average time it takes to process a single batch of examples
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
                 
    #average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'total_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'acc')

    
    #注册函数
    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.STARTED)
    def start_training(engine):
        print(1)
        engine.state.epoch = start_epoch
    
    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        print(2)
        scheduler.step()
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        #print(3)
        global ITER
        ITER += 1
        
        if ITER % log_period == 0:
            #print(5)
            logger.info("Epoch[{}] Iteration[{}/{}] Total_Loss: {:.3f}, Acc: {:.6f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['total_loss'],
                                engine.state.metrics['acc'],
                                scheduler.get_lr()[0]))

        
        if len(train_loader) == ITER:
            ITER = 0
            
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        #print(4)
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:   #每eval_period次就验证一次
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.2%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r -1]))
    
    trainer.run(train_loader, max_epochs=epochs)
