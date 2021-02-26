from imports import *

from datasets.bdd_oia import BDD_OIA_NLP

from soft_attention_model import *



def get_encoder(model_name):
    if model_name == 'mobile_net':
        md = torchvision.models.mobilenet_v2(pretrained=True)
        encoder = nn.Sequential(*list(md.children())[:-1])
    elif model_name == 'resnet':
        md = torchvision.models.resnet50(pretrained=True)
        encoder = nn.Sequential(*list(md.children())[:-2])
    return encoder



def train_one_epoch2(encoder, decoder, decoder_optimizer, data_loader, device, epoch, print_freq, encoder_optimizer=None):
    global num_iters
    encoder.train()
    decoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    for image_batch, labels_batch in metric_logger.log_every(data_loader, print_freq, header):

        if len(image_batch)!=len(labels_batch):
            continue
        
        reasons_batch = [l['reason'].to(device) for l in labels_batch]
        image_features = encoder(torch.stack(image_batch).to(device))
        loss , scores, attention_weights, hs = decoder(image_features, reasons_batch)
        num_iters += 1

        loss_value = loss.item()
        # print(loss_value)
        # reduce losses over all GPUs for logging purposes


        writer.add_scalar("Loss/train", loss_value, num_iters)
        writer.add_scalar("Decoder earning rate", decoder_optimizer.param_groups[0]["lr"], num_iters)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            sys.exit(1)

        decoder_optimizer.zero_grad()
        loss.backward()
        decoder_optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=decoder_optimizer.param_groups[0]["lr"])

    return loss_value



def get_loader(batch_size):
    ## Data loader
    image_dir = './data/bdd_oia/lastframe/data/'
    label_dir = './data/bdd_oia/lastframe/labels/'
    bdd_oia_dataset = BDD_OIA_NLP(image_dir, label_dir+'no_train.pkl', label_dir+'ind_to_word.pkl',image_min_size=180)

    training_loader = DataLoader(bdd_oia_dataset,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=0,
                                drop_last=True,
                                collate_fn=utils.collate_fn)
    NULL_INDEX = bdd_oia_dataset.word_to_ind['NULL']
    DICT_SIZE = len(bdd_oia_dataset.word_to_ind.keys())

    return training_loader,NULL_INDEX,DICT_SIZE

if __name__ == "__main__":
    encoder_model = 'resnet'
    image_f_dim = 2048  
    model_name = 'soft_attention_resnet'

    # encoder_model = 'mobile_net'
    # image_f_dim = 1280
    # model_name = 'soft_attention'

    version = 'v1'
    num_epochs = 15
    
    DEVICE = torch.device("cuda:1")
    batch_size = 10
    
    training_loader,NULL_INDEX,DICT_SIZE = get_loader(batch_size)

    writer = SummaryWriter('./runs/soft_attention/'+model_name+'/')

    encoder = get_encoder(encoder_model)
    decoder = ReasonDecoder(image_f_dim=image_f_dim,\
                            embedding_dim=128, \
                            hidden_dim=128, \
                            dict_size=DICT_SIZE, \
                            device=DEVICE,\
                            null_index=NULL_INDEX, \
                            using_gate=True)

    encoder.to(DEVICE)
    decoder.to(DEVICE)
    #### continue training 
    # checkpoint = torch.load("/home/ai/Desktop/Jiqian work/work4/saved_models/v3_hard_sel_1039.pth")
    # decision_generator.load_state_dict(checkpoint["model"])

    decoder_params = [p for p in decoder.parameters() if p.requires_grad]
    # print(len(decoder_params))
    decoder_optimizer = torch.optim.SGD(decoder_params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer)



    for epoch in tqdm(range(num_epochs)):
        # try:
        loss_value = train_one_epoch2(encoder,
                                      decoder, 
                                      decoder_optimizer, 
                                      training_loader, 
                                      DEVICE, 
                                      epoch, 
                                      print_freq=200)
        lr_scheduler.step(loss_value)
        # except Exception as e:
        #     print(e)
        
        # train_one_epoch2(decision_generator, optimizer, training_loader, device, epoch, print_freq=200)

        if (epoch+1)%5==0:
            save_name = "../saved_models/soft_attention/%s"%model_name + str(epoch) + ".pth"
            torch.save(
                {"encoder": encoder.state_dict(), 
                "decoder":decoder.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict()},
                save_name,
            )
            print("Saved model", save_name)


