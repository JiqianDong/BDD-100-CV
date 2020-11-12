from imports import *

from datasets.bdd_oia import BDD_OIA

from decision_generator_model import DecisionGenerator


device = torch.device("cuda:0")
batch_size = 10



## Data loader
image_dir = './data/bdd_oia/lastframe/data/'
label_dir = './data/bdd_oia/lastframe/labels/'


bdd_oia_dataset = BDD_OIA(image_dir,label_dir+'train_25k_images_actions.json',
                             label_dir+'train_25k_images_reasons.json')

training_loader = DataLoader(bdd_oia_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=0,
                            drop_last=True,
                            collate_fn=utils.collate_fn)



def get_model(num_classes,image_mean=bdd_oia_dataset.mean,image_std=bdd_oia_dataset.std):
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
    #                                                             image_mean=image_mean,
    #                                                             image_std=image_std)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,num_classes)
    return model



writer = SummaryWriter('./runs/sgd2/')
num_iters = 0


def train_one_epoch2(model, optimizer, data_loader, device, epoch, print_freq):
    global num_iters
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        if len(images)!=len(targets):
            continue

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        num_iters += 1
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        writer.add_scalar("Loss/train", loss_value, num_iters)
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], num_iters)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return loss_value



fastercnn = get_model(10)
checkpoint = torch.load('saved_models/bdd100k_24.pth')
fastercnn.load_state_dict(checkpoint['model'])

decision_generator = DecisionGenerator(fastercnn,device,batch_size)
decision_generator = decision_generator.to(device)
params = [p for p in decision_generator.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.Adam(params,lr=0.001, weight_decay=5e-5)
# lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=6e-3)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)



num_epochs = 70
for epoch in tqdm(range(num_epochs)):
    try:
        loss_value = train_one_epoch2(decision_generator, optimizer, training_loader, device, epoch, print_freq=200)
        lr_scheduler.step(loss_value)
    except Exception as e:
        print(e)
    
    # train_one_epoch2(decision_generator, optimizer, training_loader, device, epoch, print_freq=200)

    if (epoch+1)%10==0:
        save_name = "../saved_models/bdd_oia_sgd2" + str(epoch) + ".pth"
        torch.save(
            {"model": decision_generator.state_dict(), "optimizer": optimizer.state_dict(),},
            save_name,
        )
        print("Saved model", save_name)