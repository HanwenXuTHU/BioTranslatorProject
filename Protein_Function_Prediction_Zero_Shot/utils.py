from collections import deque
from torch.nn import init
from tqdm import tqdm
import torch
import logging
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import collections
import pickle


MEDIUM_SIZE = 8
SMALLER_SIZE = 6
BIGGER_SIZE = 25
plt.rc('font', family='Helvetica', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)# fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)# fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
FIG_HEIGHT = 2
FIG_WIDTH = 2


def save_obj(obj, name):
    '''
    Codes from https://qa.wujigu.com/qa/?qa=446145/
    '''
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    '''
    Codes from https://qa.wujigu.com/qa/?qa=446145/
    '''
    with open(name, 'rb') as f:
        return pickle.load(f)


def load(filename, with_rels=True):
    '''
    Codes Borrowed from DeepGOPlus
    '''
    ont = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    ont[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['alt_ids'] = list()
                obj['def'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                if obj is not None:
                    ont[obj['id']] = obj
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'alt_id':
                    obj['alt_ids'].append(l[1])
                elif l[0] == 'namespace':
                    obj['namespace'] = l[1]
                elif l[0] == 'def':
                    obj['def'].append(l[1].split('"')[1])
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif with_rels and l[0] == 'relationship':
                    it = l[1].split()
                    # add all types of relationships
                    obj['is_a'].append(it[1])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
        if obj is not None:
            ont[obj['id']] = obj
    for term_id in list(ont.keys()):
        for t_id in ont[term_id]['alt_ids']:
            ont[t_id] = ont[term_id]
        if ont[term_id]['is_obsolete']:
            del ont[term_id]
    for term_id, val in ont.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in ont:
                if 'children' not in ont[p_id]:
                    ont[p_id]['children'] = set()
                ont[p_id]['children'].add(term_id)
    return ont


def get_children(go, term_id):
    '''
    Codes Borrowed from DeepGOPlus
    '''
    if term_id not in go:
        return set()
    term_set = set()
    q = deque()
    q.append(term_id)
    while(len(q) > 0):
        t_id = q.popleft()
        if t_id not in term_set:
            term_set.add(t_id)
            for child_id in go[t_id]['children']:
                if child_id in go:
                    q.append(child_id)
    return term_set


def get_anchestors(go, term_id):
    '''
    Codes Borrowed from DeepGOPlus
    '''
    if term_id not in go:
        return []
    term_set = []
    q = deque()
    q.append(term_id)
    while(len(q) > 0):
        t_id = q.popleft()
        if t_id not in term_set:
            term_set.append(t_id)
            for parent_id in go[t_id]['is_a']:
                if parent_id in go:
                    q.append(parent_id)
    return term_set


def save_anchestors(save_list, go):
    a_save = collections.OrderedDict()
    for i in save_list:
        a_save[i] = get_anchestors(go, i)
    save_obj(a_save, '../data/data_2016/anchestors.pkl')
    return a_save


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Codes from Pix2Pix paper
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def vec2classes(go, terms, predictions, th=0.5, is_a=True, a_path='../data/data_2016/anchestors.pkl'):
    terms_list = np.asarray(list(terms.keys()))
    pred_classes = []
    go_term = {i: go[i] for i in list(terms.keys())}
    if is_a == False:
        anchestors = save_anchestors(list(terms.keys()), go_term)
    else:
        anchestors = load_obj(a_path)
    for i in range(np.size(predictions, 0)):
        predI = predictions[i, :]
        classI = terms_list[predI >= th]
        classA = list(terms.keys())
        x = np.asarray(list(anchestors.values()))
        t = anchestors[list(classI)]
        #for j in classI:
        #    classA += anchestors[j]
        pred_classes.append(list(set(classA)))
    debug = 0


class EarlyStopping:
    '''
    Codes from https://github.com/Bjarten/early-stopping-pytorch
    '''
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, stop_order='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_order = stop_order
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss
        if self.stop_order == 'min':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        elif self.stop_order == 'max':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Updation change ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


def get_logger(log_name='data-cafa'):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def compute_roc(labels, preds):
    '''
    Codes from DeepGOPlus
    '''
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_prc(labels, preds):
    # Compute PRC curve and PRC area for each class
    pr, rc, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    prc_auc = auc(rc, pr)
    return prc_auc


def update_model_config(model_config, file):
    model_config.emb_dim = file.emb_tensor_train.size(1)
    model_config.N_vector_dim = np.size(list(file.prot_vector.values())[0].reshape(-1), 0)
    model_config.description_dim = np.size(list(file.prot_description.values())[0].reshape(-1), 0)
    return model_config


def compute_auroc_percentage(inference_preds, inference_label, file, fold_i):
    auroc_percentage = {0.65: 0, 0.7: 0, 0.75: 0, 0.8: 0, 0.85: 0, 0.95: 0}
    for j in range(len(file.fold_zero_shot_terms_list[fold_i])):
        preds_id = inference_preds[:, j].reshape([-1, 1])
        label_id = inference_label[:, j].reshape([-1, 1])
        roc_auc_j = compute_roc(label_id, preds_id)
        for T in auroc_percentage.keys():
            if T <= roc_auc_j:
                auroc_percentage[T] += 100.0 / len(file.fold_zero_shot_terms_list[fold_i])
    return auroc_percentage


def evaluate_auroc(inference_preds, inference_label):
    roc_macro = np.zeros(np.size(inference_preds, 1))
    cl_i = 0
    for i in range(len(roc_macro)):
        unseen_preds, unseen_labels = inference_preds[:, i], inference_label[:, i]
        auroc = compute_roc(unseen_labels, unseen_preds)
        roc_macro[cl_i] = auroc
        cl_i += 1
    return np.mean(roc_macro)


def evaluate_model(model_predict, inference_dataset, file, fold_i):
    print('inferring iters')
    inference_preds, inference_label = [], []
    for j, inference_D in tqdm(enumerate(inference_dataset)):
        with torch.no_grad():
            preds = model_predict.model(inference_D['seq'], inference_D['description'], inference_D['vector'],
                                        file.emb_tensor_train)
            preds_inf, label_inf = [], []
            label = inference_D['label']
            for t_id in file.fold_zero_shot_terms_list[fold_i]:
                preds_inf.append(preds[:, file.train_terms[t_id]].reshape((-1, 1)))
                label_inf.append(label[:, file.train_terms[t_id]].reshape((-1, 1)))
            preds_inf = torch.cat(preds_inf, dim=1)
            label_inf = torch.cat(label_inf, dim=1)
            inference_preds.append(preds_inf)
            inference_label.append(label_inf)
    inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0), torch.cat(tuple(inference_label),
                                                                                           dim=0)
    inference_preds, inference_label = inference_preds.cpu().float().numpy(), inference_label.cpu().float().numpy()
    roc_auc = evaluate_auroc(inference_preds, inference_label)
    return inference_preds, inference_label, roc_auc


def auroc_bar_plot(BioTranslator_log, save_path):
    '''
    Borrowed from OnClass paper
    '''
    name_space = {'bp': 'BPO', 'mf': 'MFO', 'cc': 'CCO'}

    BioTranslator_auroc = []
    BioTranslator_err = []

    for ont in name_space.keys():
        BioTranslator_auroc.append(BioTranslator_log[ont]['mean'])
        BioTranslator_err.append(np.std(BioTranslator_log[ont]['err']) / np.sqrt(3))

    mean = np.zeros([3, 1])
    mean[:, 0] = BioTranslator_auroc


    yerr = np.zeros([3, 1])
    yerr[:, 0] = BioTranslator_err

    fig, ax = plt.subplots(figsize=(1.2*FIG_WIDTH, FIG_HEIGHT))
    n_groups = 3
    nmethod = 1
    index = np.arange(n_groups)
    bar_width = 1. / nmethod * 0.5
    opacity = 1

    ax.bar(index + (nmethod - 1) * bar_width, mean[:, 0], yerr=yerr[:, 0], width=bar_width, alpha=opacity,
           color='#66c2a5',  # ,color_l[i],
           label='BioTranslator')

    csfont = {'family': 'Helvetica'}
    ax.set_ylabel('AUROC', fontdict=csfont)
    ax.set_xticklabels(['BP', 'MF', 'CC'])
    if nmethod == 1:
        ax.set_xticks(index)
    else:
        ax.set_xticks(index + bar_width * (nmethod - 0.5) * 1. / 2 - 0.1)
    plt.legend(loc='upper right', frameon=False, ncol=1, fontsize=4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    max_y = 0.9  # min(np.ceil(np.max(mean*10))/10,1.0)
    min_y = 0.5
    if min_y > 0.70:
        min_y = 0.70
    ax.set_ylim([min_y, max_y])
    if min_y < 0.80:
        step_size = 0.
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        step_size = 0.05
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    print(min_y, max_y)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    fig.tight_layout()
    #plt.show()
    plt.savefig(save_path)


def auroc_bar_percentage_plot(percentage_BioTranslator, save_path):
    '''
    Borrowed from OnClass paper
    '''
    name_space = {'bp': 'BP', 'mf': 'MF', 'cc': 'CC'}
    fig, ax = plt.subplots(figsize=(FIG_WIDTH *4, FIG_HEIGHT * 1.4), nrows=1, ncols=3)
    fig_i = 0
    for ont in name_space.keys():
        percentage_BioTranslator_i = list(percentage_BioTranslator[ont]['mean'].values())
        percentage_BioTranslator_i_err = list(percentage_BioTranslator[ont]['err'].values())
        for i in range(len(percentage_BioTranslator_i_err)):
            percentage_BioTranslator_i_err[i] = np.std(percentage_BioTranslator_i_err[i]) / np.sqrt(3)

        y_label = ['0%', '20%', '40%', '60%', '80%', '100%']
        y_x = [0, 20, 40, 60, 80, 100]

        mean, yerr = np.zeros([len(percentage_BioTranslator_i), 1]), np.zeros([len(percentage_BioTranslator_i), 1])
        mean[:, 0] = percentage_BioTranslator_i
        yerr[:, 0] = percentage_BioTranslator_i_err

        n_groups = len(percentage_BioTranslator_i)
        nmethod = 1
        index = np.arange(n_groups)
        bar_width = 1. / nmethod * 0.8
        opacity = 1

        ax[fig_i].bar(index + (nmethod - 1) * bar_width, mean[:, 0], yerr=yerr[:, 0], width=bar_width, alpha=opacity,
               color='#66c2a5',  # ,color_l[i],
               label='BioTranslator')
        csfont = {'family': 'Helvetica'}
        ax[fig_i].set_ylabel('Percentage of functions \n AUROC > threshold', fontdict=csfont)
        ax[fig_i].set_xticklabels(['0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95'])

        if nmethod == 1:
            ax[fig_i].set_xticks(index)
        else:
            ax[fig_i].set_xticks(index + bar_width * (nmethod - 0.5) * 1. / 2 - 0.1)
        ax[fig_i].legend(loc='upper right', frameon=False, ncol=1, fontsize=4)
        # plt.legend(loc='upper left',bbox_to_anchor=(0.1, 1.1), frameon=False, ncol=1, fontsize=4)

        plt.setp(ax[fig_i].get_xticklabels(), rotation=90, ha="right", va="center",
                 rotation_mode="anchor")

        ax[fig_i].spines['right'].set_visible(False)
        ax[fig_i].spines['top'].set_visible(False)

        max_y = 100  # min(np.ceil(np.max(mean*10))/10,1.0)
        min_y = 0
        if min_y > 70:
            min_y = 70
        ax[fig_i].set_ylim([min_y, max_y])
        if min_y < 80:
            step_size = 10
            ax[fig_i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        else:
            step_size = 5
            ax[fig_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        print(min_y, max_y)
        ax[fig_i].set_yticks(y_x)
        ax[fig_i].set_yticklabels(y_label)
        # ax.legend(method_l)

        x0, x1 = ax[fig_i].get_xlim()
        y0, y1 = ax[fig_i].get_ylim()
        ax[fig_i].set_aspect(abs(x1 - x0) / abs(y1 - y0))
        ax[fig_i].set_xlabel('Threshold', fontdict=csfont)
        fig_i += 1
        fig.tight_layout()
        #plt.show()
        fig.savefig(save_path)