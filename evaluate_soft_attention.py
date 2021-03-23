from imports import *

from datasets.bdd_oia import BDD_OIA_NLP

from soft_attention_model import *

from nltk.translate.bleu_score import corpus_bleu


def get_encoder(model_name):
    if model_name == 'mobile_net':
        md = torchvision.models.mobilenet_v2(pretrained=True)
        encoder = nn.Sequential(*list(md.children())[:-1])
    elif model_name == 'resnet':
        md = torchvision.models.resnet50(pretrained=True)
        encoder = nn.Sequential(*list(md.children())[:-2])
    else:

        raise ValueError("unrecognized encoder model name: "+encoder_name)
    return encoder


def get_loader(batch_size):
    ## Data loader
    image_dir = './data/bdd_oia/lastframe/data/'
    label_dir = './data/bdd_oia/lastframe/labels/'
    bdd_oia_dataset = BDD_OIA_NLP(image_dir, label_dir+'no_test.pkl', label_dir+'ind_to_word.pkl',image_min_size=180)

    test_loader = DataLoader(bdd_oia_dataset,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=0,
                                drop_last=True,
                                collate_fn=utils.collate_fn)
    NULL_INDEX = bdd_oia_dataset.word_to_ind['NULL']
    DICT_SIZE = len(bdd_oia_dataset.word_to_ind.keys())

    return test_loader,NULL_INDEX,DICT_SIZE,bdd_oia_dataset

if __name__ == "__main__":

    DEVICE = torch.device("cpu")

    encoder_name,image_f_dim,model_name,using_gate = 'mobile_net',1280, "soft_attention14", True 
    # encoder_name,image_f_dim,model_name,using_gate = 'resnet',2048, "soft_attention_resnet14", True
    # encoder_name,image_f_dim,model_name,using_gate = 'resnet',2048, "soft_attention_resnet_nogate14",False

    test_loader,NULL_INDEX,DICT_SIZE,bdd_oia_dataset = get_loader(10)
    encoder = get_encoder(encoder_name)


    ind_to_word = bdd_oia_dataset.ind_to_word
    word_to_ind = bdd_oia_dataset.word_to_ind


    decoder = ReasonDecoder(image_f_dim=image_f_dim,\
                            embedding_dim=128, \
                            hidden_dim=128, \
                            dict_size=DICT_SIZE, \
                            device='cpu',\
                            null_index=NULL_INDEX, \
                            using_gate=using_gate)

    checkpoint = torch.load('./saved_models/soft_attention/'+model_name+'.pth')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    decoder.eval()

    encoder.to(DEVICE)
    decoder.to(DEVICE)

    total_reason_predicted = []
    total_reason_label = []

    for batch_num,(images_batch,labels_batch) in enumerate(tqdm(test_loader)):
        image_batch = torch.stack(images_batch).to(DEVICE)
        reason_batch = [l['reason'] for l in labels_batch]
        reasons = [[ind_to_word[i] for i in rb.numpy()] for rb in reason_batch]
        total_reason_label.extend(reasons)

        image_f = encoder(image_batch)
        predictions = decoder.generate_reason(image_f)
        pred_batch = predictions[-1].T.detach().cpu()  # B * T

        for p in pred_batch:
            tmp = []
            for i in p.numpy():
                tmp.append(ind_to_word[i])
                if ind_to_word[i]=='EOS':
                    break
            total_reason_predicted.append(tmp)
        # break

    assert len(total_reason_predicted) == len(total_reason_label)

    with open('./saved_predictions/soft_attention/'+model_name+'_pred.pkl','wb') as f:
        pickle.dump(total_reason_predicted,f)
    with open('./saved_predictions/soft_attention/'+model_name+'_label.pkl','wb') as f:
        pickle.dump(total_reason_label,f)

    bleu = corpus_bleu(total_reason_label,total_reason_predicted,(1,0,0,0))
    print("bleu score is: ", bleu)




