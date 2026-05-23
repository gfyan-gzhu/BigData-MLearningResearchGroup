import logging
import os.path
import time


logger = logging.getLogger()

def creat_log():
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def save_log(exp_tag=''):

    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    if exp_tag:
        log_filename = f'{timestamp}_{exp_tag}.log'
    else:
        log_filename = f'{timestamp}.log'
    log_filepath = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f'日志文件已创建: {log_filepath}')
    logger.info('=' * 50)
    logger.info('LcDF蒸馏开始')
    logger.info('=' * 50)

def log_experiment_config(config):
    logger.info('实验核心配置:')
    logger.info('-' * 50)
    for key, value in config.items():
        logger.info(f'{key}: {value}')
    logger.info('-' * 50)
    logger.info('训练进度:')
    logger.info('Epoch | Train Loss | Val Loss | Val Acc | Gamma')
    logger.info('-' * 70)

def log_epoch_result(epoch, train_loss, val_loss, val_acc, gamma):
    logger.info(
        f'{epoch:5d} | {train_loss:.4f} | {val_loss:.4f} | {val_acc:.4f} | {gamma:.4f}'
    )

def log_final_result(best_acc, best_epoch, total_time):
    logger.info('-' * 70)
    logger.info('实验结束！')
    logger.info(f'最高验证准确率: {best_acc:.4f} (第{best_epoch}轮)')
    logger.info(f'总训练时间: {total_time / 3600:.2f} 小时')
    logger.info('=' * 50)