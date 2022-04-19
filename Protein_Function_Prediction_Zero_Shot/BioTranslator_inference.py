import torch
import collections
from model import BioTranslatorModel
from options import model_config, data_loading, inference_config
from file_loader import FileLoader
from torch.utils.data import DataLoader
from utils import init_weights, EarlyStopping, compute_roc, auroc_bar_plot, auroc_bar_percentage_plot, update_model_config
from tqdm import tqdm
import numpy as np
import pickle


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


class BioTranslator:

    def __init__(self, model_config):
        self.loss_func = torch.nn.BCELoss()
        self.model = BioTranslatorModel.BioTranslatorModel(input_nc=model_config.input_nc,
                                                in_nc=model_config.in_nc,
                                                max_kernels=model_config.max_kernels,
                                                hidden_dim=model_config.hidden_dim,
                                                feature=model_config.features,
                                                seqL=model_config.max_len,
                                                emb_dim=model_config.emb_dim)
        if len(model_config.gpu_ids) > 0:
            init_weights(self.model, init_type='xavier')
            self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.lr)

    def backward_model(self, input_seq, input_description, input_vector, emb_tensor, label):
        preds = self.model(input_seq, input_description, input_vector, emb_tensor)
        self.loss = self.loss_func(preds, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


def evaluate_auroc(inference_preds, inference_label):
    roc_macro = np.zeros(np.size(inference_preds, 1))
    cl_i = 0
    for i in range(len(roc_macro)):
        unseen_preds, unseen_labels = inference_preds[:, i], inference_label[:, i]
        auroc = compute_roc(unseen_labels, unseen_preds)
        roc_macro[cl_i] = auroc
        cl_i += 1
    return np.mean(roc_macro)


def main():
    data_opt = data_loading()
    model_opt = model_config()
    infer_opt = inference_config()
    torch.cuda.set_device('cuda:' + data_opt.gpu_ids)
    print('dataset is {}'.format(data_opt.dataset))
    file = FileLoader(data_opt)
    print('loading data finished')
    name_space = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}

    auroc_ont = collections.OrderedDict()
    auroc_percentage_ont = collections.OrderedDict()
    for ont in ['bp', 'mf', 'cc']:
        auroc_total = collections.OrderedDict()
        auroc_total['mean'] = 0
        auroc_total['err'] = []
        auroc_percentage = {0.65: 0, 0.7: 0, 0.75: 0, 0.8: 0, 0.85: 0, 0.9: 0, 0.95: 0}
        auroc_percentage_err = {0.65: [], 0.7: [], 0.75: [], 0.8: [], 0.85: [], 0.9: [], 0.95: []}
        for T_i in auroc_percentage_err.keys():
            for k in range(file.k_fold):
                auroc_percentage_err[T_i].append(0)
        for fold_i in range(3):
            model_opt = update_model_config(model_opt, file)
            model_predict = torch.load(model_opt.save_path.format(fold_i))
            model_predict.model.eval()
            inference_dataset = DataLoader(file.fold_validation[fold_i], batch_size=model_opt.batch_size, shuffle=True)

            print('Ont:{} Fold:{} inferring iters'.format(ont, fold_i))
            inference_preds, inference_label = [], []
            zero_shot_ont_terms = []
            for t_id in file.fold_zero_shot_terms_list[fold_i].keys():
                if file.go_data[t_id]['namespace'] == name_space[ont]:
                    zero_shot_ont_terms.append(t_id)
            for j, inference_D in tqdm(enumerate(inference_dataset)):
                with torch.no_grad():
                    preds = model_predict.model(inference_D['seq'], inference_D['description'], inference_D['vector'], file.emb_tensor_train)
                    preds_inf, label_inf = [], []
                    label = inference_D['label']
                    for t_id in zero_shot_ont_terms:
                        preds_inf.append(preds[:, file.train_terms[t_id]].reshape((-1, 1)))
                        label_inf.append(label[:, file.train_terms[t_id]].reshape((-1, 1)))
                    preds_inf = torch.cat(preds_inf, dim=1)
                    label_inf = torch.cat(label_inf, dim=1)
                    inference_preds.append(preds_inf)
                    inference_label.append(label_inf)
            inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0), torch.cat(tuple(inference_label), dim=0)
            inference_preds, inference_label = inference_preds.cpu().float().numpy(), inference_label.cpu().float().numpy()
            roc_auc = evaluate_auroc(inference_preds, inference_label)
            auroc_total['mean'] += roc_auc / file.k_fold
            auroc_total['err'].append(roc_auc)

            #compute auroc percentage:
            for j in range(len(zero_shot_ont_terms)):
                preds_id = inference_preds[:, j].reshape([-1, 1])
                label_id = inference_label[:, j].reshape([-1, 1])
                roc_auc_j = compute_roc(label_id, preds_id)
                for T in auroc_percentage.keys():
                    if T <= roc_auc_j:
                        auroc_percentage[T] += 100.0/len(zero_shot_ont_terms)/file.k_fold
                        auroc_percentage_err[T][fold_i] += 100.0/len(zero_shot_ont_terms)
        auroc_ont[ont] = auroc_total
        auroc_percentage_ont[ont] = collections.OrderedDict()
        auroc_percentage_ont[ont]['mean'] = auroc_percentage
        auroc_percentage_ont[ont]['err'] = auroc_percentage_err

    save_obj(auroc_ont, infer_opt.save_auroc_path) # where you store the inference results
    save_obj(auroc_percentage_ont, infer_opt.save_auroc_geq_threshold_percentage) # where you store the inference results
    # draw auroc bar plot
    auroc_bar_plot(auroc_ont, infer_opt.barplot_save_path)
    auroc_bar_percentage_plot(auroc_percentage_ont, infer_opt.auroc_geq_threshold_barplot_save_path)



if __name__ == '__main__':
    main()