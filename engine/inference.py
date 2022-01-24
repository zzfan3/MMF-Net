import logging

import torch
import torch.nn as nn
import torch.utils.data
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking
from utils.logger_result import setup_logger

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
    
def inference(cfg, model, val_loader, num_query, output_dir):
    device = cfg.MODEL.DEVICE

    logger = setup_logger("reid_baseline.inference", output_dir, 0)
   #logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Creat evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))
    
    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    