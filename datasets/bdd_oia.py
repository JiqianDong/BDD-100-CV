from imports import *

class BDD_OIA(Dataset):
    def __init__(self, imageRoot, gtRoot, reasonRoot):
        super().__init__()

        self.mean=[102.9801, 115.9465, 122.7717]
        self.std=[1., 1., 1.]

        self._processing(imageRoot, gtRoot, reasonRoot)

        self.transform = T.Compose([T.ToTensor()])

    def _processing(self,imageRoot, gtRoot, reasonRoot):
        with open(gtRoot) as json_file:
            data = json.load(json_file)
        with open(reasonRoot) as json_file:
            reason = json.load(json_file)

        data['images'] = sorted(data['images'], key=lambda k: k['file_name'])
        reason = sorted(reason, key=lambda k: k['file_name'])
        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            # print(len(action_annotations[ind]['category']))
            if len(action_annotations[ind]['category']) == 4  or action_annotations[ind]['category'][4] == 0:
                file_name = os.path.join(imageRoot, img['file_name'])
                if os.path.isfile(file_name):
                    self.imgNames.append(file_name)
                    self.targets.append(torch.FloatTensor(action_annotations[ind]['category']))
                    self.reasons.append(torch.FloatTensor(reason[i]['reason']))

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # test = True
        imgName = self.imgNames[idx]
        target = {}
        target['action'] = self.targets[idx][:4]
        target['reason'] = self.reasons[idx]

        img_ = Image.open(imgName)

        img,tgt = self.transform(img_,target)

        return img,tgt
