from imports import *

from datasets.bdd_oia import BDD_OIA

from decision_generator_model import *

from sklearn.metrics import f1_score



def main(model_name,version=None,sel_k=10):
    device = torch.device("cuda:0")
    batch_size = 10
    ## Data loader
    image_dir = './data/bdd_oia/lastframe/data/'
    label_dir = './data/bdd_oia/lastframe/labels/'

    # bdd_oia_dataset = BDD_OIA(image_dir,label_dir+'train_25k_images_actions.json',
    #                              label_dir+'train_25k_images_reasons.json')

    # training_loader = DataLoader(bdd_oia_dataset,
    #                             shuffle=True,
    #                             batch_size=batch_size,
    #                             num_workers=0,
    #                             drop_last=True,
    #                             collate_fn=utils.collate_fn)

    val_bdd_oia_dataset = BDD_OIA(image_dir,label_dir+'val_25k_images_actions.json',
                                label_dir+'val_25k_images_reasons.json')

    val_loader = DataLoader(val_bdd_oia_dataset,shuffle=False,batch_size=batch_size,collate_fn=utils.collate_fn)

    batch_size = 10

    def get_model(num_classes):
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
        #                                                             image_mean=image_mean,
        #                                                             image_std=image_std)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,num_classes)
        return model


    fastercnn = get_model(10)
    checkpoint = torch.load('saved_models/bdd100k_24.pth')
    fastercnn.load_state_dict(checkpoint['model'])

    if version=='v3':
        decision_generator = DecisionGenerator_v3(fastercnn,device,batch_size, select_k=sel_k)
    elif version == 'v1':
        decision_generator = DecisionGenerator_v1(fastercnn,device,batch_size)
    elif version == 'v4':
        decision_generator = DecisionGenerator_v4(fastercnn,device, batch_size)
    else:
        ############# Load version 2 ###########################
        decision_generator = DecisionGenerator(fastercnn,device,batch_size, select_k=sel_k)
    decision_generator = decision_generator.to(device)

    checkpoint = torch.load("/home/ai/Desktop/Jiqian work/work4/saved_models/%s.pth"%model_name)
    decision_generator.load_state_dict(checkpoint["model"])
    

    ############# Load version 1 ###########################
    # from decision_generator_model import DecisionGenerator_v1
    # decision_generator = DecisionGenerator_v1(fastercnn,device,batch_size)

    # checkpoint = torch.load("saved_models/bdd_oia19.pth")
    # decision_generator.load_state_dict(checkpoint["model"])


    ############################################################

    decision_generator = decision_generator.to(device)
    decision_generator.eval()

    count = val_loader.__len__()
    overall_pred_action = []
    overall_pred_reason = []
    overall_trgt_action = []
    overall_trgt_reason = []
    for i,databatch in enumerate(val_loader):
        print('Finished: {} / {}'.format(i, count))
        print('Finished: %.2f%%' % (i /count * 100))
        torch.cuda.empty_cache()

        images, targets = databatch
        images_cuda = list(image.to(device) for image in images)

        pred = decision_generator(images_cuda)

        pred_action = (pred['action'].detach().cpu().numpy()>0.5)*1
        pred_reason = (pred['reasons'].detach().cpu().numpy()>0.5)*1

        target_action = np.stack([t['action'].numpy() for t in targets])
        target_reason = np.stack([t['reason'].numpy() for t in targets])

        overall_pred_action.append(pred_action)
        overall_pred_reason.append(pred_reason)
        overall_trgt_action.append(target_action)
        overall_trgt_reason.append(target_reason)


    overall_pred_action = np.vstack(overall_pred_action)
    overall_pred_reason = np.vstack(overall_pred_reason)
    overall_trgt_action = np.vstack(overall_trgt_action)
    overall_trgt_reason = np.vstack(overall_trgt_reason)


    action_f1 = f1_score(overall_pred_action, overall_trgt_action, average='samples')
    reason_f1 = f1_score(overall_pred_reason, overall_trgt_reason, average='samples')

    print('action f1: ', action_f1)
    print('reason_f1: ', reason_f1)


    with open('./saved_predictions/pred_action_' + model_name +'.npy','wb') as f:
        np.save(f, overall_pred_action)

    with open('./saved_predictions/pred_reason_'+ model_name +'.npy','wb') as f:
        np.save(f, overall_pred_reason)

    with open('./saved_predictions/trgt_action_' + model_name +'.npy','wb') as f:
        np.save(f, overall_trgt_action)

    with open('./saved_predictions/trgt_reason_'+ model_name +'.npy','wb') as f:
        np.save(f, overall_trgt_reason)


if __name__ == "__main__":
    # model_name = 'v3_hard_sel_1039'
    # version = 'v3'
    model_name = 'v4_mhsa_test39'
    version = 'v4'
    sel_k = 10
    main(model_name,version)