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

        print(len(self.reasons),len(self.targets),len(self.imgNames))
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



class BDD_SUBSET(Dataset):
    def __init__(self, image_root, label_root, ind_to_word_root):
        super().__init__()
        import torchvision.transforms as T

        self.image_root = image_root
        self.mean=[102.9801, 115.9465, 122.7717]
        self.std=[1., 1., 1.]

        self._processing(label_root)

        self.transform = T.Compose([T.ToTensor(),
                                   T.Normalize(self.mean,self.std)])

    def _processing(self, label_root, label_dic_root):
        data_df = pd.read_csv(label_root)

        self.count = len(data_df)
        self.all_images = data_df['file_name']
        self.all_reasons = data_df['reason_lang_ind']
        self.all_actions = data_df['action']
        print("number of samples in dataset:{}".format(self.count))

        with open(label_dic_root,'rb') as f:
            self.ind_to_word = pickle.load(f)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # test = True
        target = {}
        image_name = self.all_images[idx]
        target['action'] = self.all_actions[idx][:4]
        target['reason'] = self.all_reasons[idx]

        img_ = Image.open(self.image_root + image_name)

        img,tgt = self.transform(img_,target)

        return img,tgt