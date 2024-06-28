import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import random
from os import walk
import glob
import pickle

import warnings
warnings.filterwarnings("ignore")


class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all images
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, 
    especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize,
                 expr: str, startidx=0, ):
        """
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        self.mode = mode
        self.expr = expr
        
        if expr not in ['regular', 'bml', 'regular+bml']:
            raise ValueError("Invalid experiment type. Try regular, bml or regular+bml")
            
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, mode)  # image path
        
        # :return: dictLabels: {label1: [filename1, filename2, filename3, filename4,...], }
        dictLabels = self.loadCSV(root, mode)  # csv path
        dictLabels1 = self.loadCSV1(root, mode)
        #print(dictLabels1)
        self.data = []
        self.img2label = {}
        for i, (label, imgs) in enumerate(dictLabels.items()):
            self.data.append(imgs)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[label] = i + self.startidx  # {"img_name[:9]":label}
        self.cls_num = len(self.data)
        
        self.data1 = []
        self.img2label1 = {}
        for i, (label1, imgs1) in enumerate(dictLabels1.items()):
            self.data1.append(imgs1)  # [[img1, img2, ...], [img111, ...]]
            self.img2label1[label1] = i + self.startidx  # {"img_name[:9]":label}
        self.cls_num1 = len(self.data1)

        self.create_batch(self.batchsz)
        self.create_batch1(self.batchsz)
         
            
        if self.expr == "bml":
            self.support_x_batch = self.support_x_batch 
            self.query_x_batch = self.query_x_batch 
            self.selected_classes = self.selected_classes 
        elif self.expr == "bml+regular":
            self.support_x_batch = self.support_x_batch +  self.support_x_batch1
            self.query_x_batch = self.query_x_batch +  self.query_x_batch1
            self.selected_classes = self.selected_classes +  self.selected_classes1
        elif self.expr == "regular":
            self.support_x_batch = self.support_x_batch1
            self.query_x_batch = self.query_x_batch1
            self.selected_classes = self.selected_classes1
            

    def loadCSV(self, root, mode):
        
        if mode == 'train':
            with open(root+"final_pos_classes_train_10.pkl", "rb") as fp:
                file = pickle.load(fp)
            
            file_dict = {v: k for v, k in enumerate(file)}
            
            with open(root+"final_neg_classes_train_10.pkl", "rb") as fp:
                file_neg = pickle.load(fp)
            
            file_neg_dict = {v+len(file_dict): k for v, k in enumerate(file_neg)}
            
            def Merge(dict1, dict2):
                res = {**dict1, **dict2}
                return res
            
            dictLabels = Merge(file_dict, file_neg_dict)
            
        elif mode == 'test':

            with open(root+"final_pos_classes_test_10.pkl", "rb") as fp:
                file = pickle.load(fp)
            
            file_dict = {v: k for v, k in enumerate(file)}
            
            with open(root+"final_neg_classes_test_10.pkl", "rb") as fp:
                file_neg = pickle.load(fp)
            
            file_neg_dict = {v+len(file_dict): k for v, k in enumerate(file_neg)}
            
            def Merge(dict1, dict2):
                res = {**dict1, **dict2}
                return res
            
            dictLabels = Merge(file_dict, file_neg_dict)

        return dictLabels
    
    def loadCSV1(self, root, mode):
        
        mode = mode+'/'
        path = os.path.join(root, mode) 
        
        filenames = next(walk(path))[1]
    
        dictLabels = {}
        
        for i in range(len(filenames)):  
            img = []
            for images in glob.iglob(f'{path+filenames[i]}/*'):
                # check if the image ends with png
                if (images.endswith(".jpg")) or (images.endswith(".JPEG")):
                    img_temp = images[len(path+filenames[i]+'/'):]
                    img_temp = filenames[i]+'/'+img_temp
                    img.append(img_temp)
                
                dictLabels[filenames[i]] = img
                
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        self.selected_classes = []
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls_pos = np.random.choice(int(self.cls_num/2), 1, False)  # no duplicate
            selected_cls_neg = selected_cls_pos+int(self.cls_num/2) # no duplicate
            selected_cls = np.concatenate((selected_cls_pos, selected_cls_neg), axis=0)
            support_x = []
            query_x = []
            selected_classes_temp = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                #print(selected_imgs_idx)
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())
                selected_classes_temp.append(cls)

            # shuffle the correponding relation between support set and query set
            # random.shuffle(support_x)
            # random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            #print(self.support_x_batch)
            self.query_x_batch.append(query_x)  # append sets to current sets
            self.selected_classes.append(selected_classes_temp)
   
    def create_batch1(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch1 = []  # support set batch
        self.query_x_batch1 = []  # query set batch
        self.selected_classes1 = []
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls1 = np.random.choice(self.cls_num1, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls1)
            support_x1 = []
            query_x1 = []
            selected_classes_temp1 = []
            for cls in selected_cls1:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx1 = np.random.choice(len(self.data1[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx1)
                indexDtrain1 = np.array(selected_imgs_idx1[:self.k_shot])  # idx for Dtrain
                indexDtest1 = np.array(selected_imgs_idx1[self.k_shot:])  # idx for Dtest
                support_x1.append(
                    np.array(self.data1[cls])[indexDtrain1].tolist())  # get all images filename for current Dtrain
                query_x1.append(np.array(self.data1[cls])[indexDtest1].tolist())
                selected_classes_temp1.append(cls)

            # shuffle the correponding relation between support set and query set
            # random.shuffle(support_x1)
            # random.shuffle(query_x1)

            self.support_x_batch1.append(support_x1)  # append set to current sets
            self.query_x_batch1.append(query_x1)  # append sets to current sets
            #print(self.query_x_batch1)
            self.selected_classes1.append(selected_classes_temp1)

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        #support_y = np.zeros((self.setsz), dtype=np.int32)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        #query_y = np.zeros((self.querysz), dtype=np.int32)

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        # support_y = np.array(
        #     [self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
        #      for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)
        
        support_y_list = []
        for i in range(len(self.support_x_batch[index])):
            class_temp = np.repeat(self.selected_classes[index][i], len(self.support_x_batch[index][i]))
            support_y_list.append(class_temp)
        support_y = np.array(support_y_list).flatten().astype(np.int32)

        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        # query_y = np.array([self.img2label[item[:9]]
        #                     for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
        
        query_y_list = []
        for i in range(len(self.query_x_batch[index])):
            class_temp = np.repeat(self.selected_classes[index][i], len(self.query_x_batch[index][i]))
            query_y_list.append(class_temp)
        query_y = np.array(query_y_list).flatten().astype(np.int32)

        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
            
        #print(len(support_x))

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)
    
    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz
