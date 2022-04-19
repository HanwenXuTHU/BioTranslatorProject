import options
import torch
from tqdm import tqdm
from dataloader import KNNData, src2tgtData
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from model import Textomics
from utils import get_logger, sentences_evaluation, save_obj, save_test_text


def main():
    opt = options.options()
    knn_data = KNNData(opt)
    logger = get_logger(opt.logger_name)

    for fold_i in range(opt.nFold):

        textomics_ins = Textomics(opt).to(opt.device_0)
        optimizer = AdamW(textomics_ins.parameters(), lr=opt.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    opt.num_warmup_steps,
                                                    opt.epoch * 1200)

        train_knn, test_knn = knn_data.generate_data(random_state=fold_i)
        train_knn, test_knn = src2tgtData(train_knn), src2tgtData(test_knn)
        train_loader = DataLoader(train_knn, batch_size=opt.batch_size, shuffle=True)
        test_loader = DataLoader(test_knn, batch_size=opt.batch_size, shuffle=True)
        for i in range(opt.epoch):
            overall_loss = 0
            num = 0
            print('Training ......\n')
            for j, train_batch in tqdm(enumerate(train_loader)):
                src_text, tgt_text, src_dist = train_batch['source'], train_batch['target'], train_batch['dist']
                optimizer.zero_grad()
                outputs, loss = textomics_ins(src_text, tgt_text, dist=src_dist)
                loss.backward()
                optimizer.step()
                scheduler.step()
                overall_loss += loss.item()
                num += 1
            print('epoch: {} training loss: {}'.format(i, overall_loss / num))

            if i > 0:
                if i % opt.eval_interval == 0 or i == opt.epoch - 1:
                    preds, labels, nst_preds = [], [], []
                    with torch.no_grad():
                        print('Inferring ......\n')
                        for j, test_batch in tqdm(enumerate(test_loader)):
                            src_text, tgt_text, src_dist = test_batch['source'], test_batch['target'], test_batch['dist']
                            pred_text = textomics_ins.generate(src_text, dist=src_dist)
                            preds += list(pred_text)
                            labels += list(tgt_text)
                            nst_preds += list(src_text[0])
                        bleu_s, rouge_s, meteor, nist, all_metrics = sentences_evaluation(preds, labels)
                        logger.info('Fold:{} epoch:{}'.format(fold_i, i)
                                    + ' bleu:{}, bleu1:{}, bleu2:{}, bleu3:{}, bleu4:{}'
                                    .format(bleu_s['bleu_avg'], bleu_s['bleu1_avg'], bleu_s['bleu2_avg'], bleu_s['bleu3_avg'], bleu_s['bleu4_avg']))
                        logger.info('Fold:{} epoch:{}'.format(fold_i, i)
                                    + ' rouge1:{}, rouge2:{}, rougeL:{}'
                                    .format(rouge_s['rouge1_avg'], rouge_s['rouge2_avg'], rouge_s['rougeL_avg']))
                        logger.info('Fold:{} epoch:{}'.format(fold_i, i)
                                    + ' meteor:{}, nist:{}'
                                    .format(meteor, nist))

                        bleu_s, rouge_s, meteor, nist, nst_all_metrics = sentences_evaluation(nst_preds, labels)
                        logger.info('Fold:{} epoch:{}'.format(fold_i, i)
                                    + ' nearest bleu:{}, bleu1:{}, bleu2:{}, bleu3:{}, bleu4:{}'
                                    .format(bleu_s['bleu_avg'], bleu_s['bleu1_avg'], bleu_s['bleu2_avg'],
                                            bleu_s['bleu3_avg'], bleu_s['bleu4_avg']))
                        logger.info('Fold:{} epoch:{}'.format(fold_i, i)
                                    + ' nearest rouge1:{}, rouge2:{}, rougeL:{}'
                                    .format(rouge_s['rouge1_avg'], rouge_s['rouge2_avg'], rouge_s['rougeL_avg']))
                        logger.info('Fold:{} epoch:{}'.format(fold_i, i)
                                    + ' nearest meteor:{}, nist:{}'
                                    .format(meteor, nist))

                    save_test_text(preds, nst_preds, labels, opt.save_test_text.format(fold_i, i))
        save_obj(all_metrics, opt.save_metrics.format(fold_i))
        save_obj(nst_all_metrics, opt.nst_metrics.format(fold_i))
        torch.save(textomics_ins, opt.save_model.format(fold_i))





if __name__ == '__main__':
    main()