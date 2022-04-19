from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig


def main():
    model_name = "allenai/specter"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer.save_pretrained('../../hugging_face_model/' + model_name + '/')
    model.save_pretrained('../../hugging_face_model/' + model_name + '/')
    #model = AutoModel.from_pretrained('/home/hwxu/bert_save/' + eT_name + '_tokenizer/')


if __name__ == '__main__':
    main()