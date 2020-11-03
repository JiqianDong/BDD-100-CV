from imports import *

from datasets.bdd_oia import BDD_OIA

from decision_generator_model import DecisionGenerator


device = torch.device("cuda:0")
batch_size = 10


from datasets.bdd_oia import BDD_OIA

## Data loader
image_dir = './data/bdd_oia/lastframe/data/'
label_dir = './data/bdd_oia/lastframe/labels/'


bdd_oia_dataset = BDD_OIA(image_dir,label_dir+'train_25k_images_actions.json',
                             label_dir+'train_25k_images_reasons.json')

training_loader = DataLoader(bdd_oia_dataset,shuffle=True,batch_size=batch_size,num_workers=0,collate_fn=utils.collate_fn)



def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,num_classes)
    return model

fastercnn = get_model(10)
checkpoint = torch.load('saved_models/bdd100k_24.pth')
fastercnn.load_state_dict(checkpoint['model'])

decision_generator = DecisionGenerator(fastercnn,device,batch_size)
decision_generator = decision_generator.to(device)
params = [p for p in decision_generator.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=6e-3)


num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    try:
        train_one_epoch(decision_generator, optimizer, training_loader, device, epoch, print_freq=200)
    except Exception as e:
        print(e)
    lr_scheduler.step()

    if (epoch+1)%5==0:
        save_name = "../saved_models/bdd_oia" + str(epoch) + ".pth"
        torch.save(
            {"model": decision_generator.state_dict(), "optimizer": optimizer.state_dict(),},
            save_name,
        )
        print("Saved model", save_name)