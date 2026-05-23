import argparse
import datetime
import os

import yaml
import torch
import time
import numpy as np
from collections import defaultdict, OrderedDict

from src.model_handler import ModelHandler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config):
    print_config(config)
    set_random_seed(config['seed'])
    model_handler = ModelHandler(config)
    # model.to(device)  # 将模型移动到设备上
    f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model_handler.train()
    print("F1-Macro: {}".format(f1_mac_test))
    print("AUC: {}".format(auc_test))
    print("G-Mean: {}".format(gmean_test))
    # Visualize embeddings after training
    try:
        # Create visualization directory if it doesn't exist
        viz_dir = os.path.join(config['save_dir'], 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        # Generate visualization filename
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        viz_filename = f"node_embeddings_{config['data_name']}_{config['model']}_{timestamp}.png"
        viz_path = os.path.join(viz_dir, viz_filename)

        # Generate and save visualization
        print("\nGenerating node embeddings visualization...")
        model_handler.visualize_embeddings(save_path=viz_path)
        print("Visualization completed!")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")


def multi_run_main(config):
    print_config(config)
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    f1_list, f1_1_list, f1_0_list, auc_list, gmean_list = [], [], [], [], []
    configs = grid(config)
    for i, cnf in enumerate(configs):
        print('Running {}:\n'.format(i))
        for k in hyperparams:
            cnf['save_dir'] += '{}_{}_'.format(k, cnf[k])
        print(cnf['save_dir'])
        set_random_seed(cnf['seed'])
        st = time.time()
        model = ModelHandler(cnf)
        f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
        f1_list.append(f1_mac_test)
        f1_1_list.append(f1_1_test)
        f1_0_list.append(f1_0_test)
        auc_list.append(auc_test)
        gmean_list.append(gmean_test)
        print("Running {} done, elapsed time {}s".format(i, time.time()-st))
        
    print("F1-Macro: {}".format(f1_list))
    print("AUC: {}".format(auc_list))
    print("G-Mean: {}".format(gmean_list))

    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list, ddof=1)
    f1_1_mean, f1_1_std = np.mean(f1_1_list), np.std(f1_1_list, ddof=1)
    f1_0_mean, f1_0_std = np.mean(f1_0_list), np.std(f1_0_list, ddof=1)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list, ddof=1)
    gmean_mean, gmean_std = np.mean(gmean_list), np.std(gmean_list, ddof=1)

    print("F1-Macro: {}+{}".format(f1_mean, f1_std))
    print("F1-binary-1: {}+{}".format(f1_1_mean, f1_1_std))
    print("F1-binary-0: {}+{}".format(f1_0_mean, f1_0_std))
    print("AUC: {}+{}".format(auc_mean, auc_std))
    print("G-Mean: {}+{}".format(gmean_mean, gmean_std))



################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    if cfg['multi_run']:
        multi_run_main(config)
    else:
        main(config)
