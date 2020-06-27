import logging, os, random
import torch, numpy as np

def set_logging_defaults(args):
    os.makedirs(args.logdir)

    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])

    # log cmdline argumetns
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

def log_loss(engine):
    logger = logging.getLogger(engine.name)
    logger.info('[iteration {}] [loss {:.4f}]'.format(engine.state.iteration, engine.state.output['loss']))

def log_metrics(engine, writer, base_engine):
    logger = logging.getLogger(engine.name)
    try:
        iteration = base_engine.state.iteration
        log = '[iteration {}]'.format(iteration)
    except:
        log = '[iteration 0]'
    for name, value in engine.state.metrics.items():
        log += ' [{} {:.4f}]'.format(name, value)
        if writer is not None:
            writer.add_scalar('metrics/{}/{}'.format(engine.name, name), value, iteration)
    logger.info(log)
