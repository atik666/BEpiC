import os
from os import walk
import glob
import numpy as np
import cv2
from tqdm import tqdm
import copy
from sklearn.decomposition import PCA
import pickle
import time

# class for generating the data
class BEpiC():
    
    def __init__(self, root, mode, cluster_num, img_size = 84,
                 save_path = "/mnt/d/OneDrive - Oklahoma A and M System/RA/Fall 23/BMAML/Code/"): # main dir, data mode, cluster to be considered
    
        self.start = time.time()
    
        self.path = os.path.join(root, mode, '')  # actual dir
        self.cluster_num = cluster_num # number of cluster considered
        
        self.img_size = img_size 
        self.cluster_num = cluster_num
        self.mode = mode
        self.save_path = save_path
        
        self.file = self.load_files(self.path)  # selected file path from load_files function
        self.classes = copy.deepcopy(self.file) # copy of file
        self.images = self.images(self.file) # image array from selected file path

        # all selected neg img path, all selected neg img array, neg temp img all 
        self.neg_clusters, self.neg_images = self.neg_cluster() 
        # postive representive image index
        
        try:
            self.indx = self.rep_image(self.images, self.classes)
        except ValueError as ve:
            # Handling the caught exception and raising another exception with a custom error message
            print(f"Caught exception: {ve}")
            raise RuntimeError("Maybe you need to check the image extension.") from ve
      
    # loading postive file path          
    def load_files(self, path: str) -> list:
        
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

        return list(dictLabels.values())
    
    # loading postive img array
    def images(self, file):
        class_num = len(self.file)

        images = []
        for k in tqdm(range(class_num)):
            images_n = []
            for j in range(len(self.file[k])):
                images_temp = cv2.imread(self.path+self.file[k][j])
                images_temp = cv2.resize(images_temp, (self.img_size,self.img_size))
                images_n.append(images_temp)

            images.append(images_n)
        
        return images
    
    # selecting neg image cluster
    def neg_cluster(self):
        neg_clusters = []
        neg_images = []
        with tqdm(total=len(self.classes)*self.cluster_num*len(self.file[0])) as pbar:
            for current_class in tqdm(range(len(self.classes))):
                file_new = copy.deepcopy(self.file)
                # random.shuffle(file_new)
                class_array_cluster = []
                image_array_cluster = []
                for j in tqdm(range(self.cluster_num)):
                    class_array = []
                    image_temp = []
                    for i in range(len(self.file[0])):
                        rand_class = int(np.random.randint(len(file_new),size=(1, 1)))
                        
                        while rand_class == current_class or len(file_new[rand_class]) == 0: # TODO: will cause problem for a smaller class number
                            rand_class = int(np.random.randint(len(file_new),size=(1, 1)))
                        
                        rand_image = int(np.random.randint(len(file_new[rand_class]),size=(1, 1)))
                
                        file_array = file_new[rand_class][rand_image]
                        file_new[rand_class].remove(file_new[rand_class][rand_image])
                        class_array.append(file_array)
                        
                        img_temp = cv2.imread(self.path+file_array)
                        img_temp = cv2.resize(img_temp, (self.img_size,self.img_size))
                        
                        image_temp.append(img_temp)
                        pbar.update(1)
                        
                    class_array_cluster.append(class_array)
                    image_array_cluster.append(image_temp)
                    
                neg_clusters.append(class_array_cluster)
                neg_images.append(image_array_cluster)
                
        return neg_clusters, neg_images
    
    # representive image for a class
    def rep_image(self, images: list, classes: list) -> list:
        
        # normalize
        def normalizeData(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        
        indexes = []
        for j in tqdm(range(len(self.images))):
            
            img2 =  [self.images[j][i].mean(axis=2).flatten() for i in range(len(self.images[j]))]
            
            img2 = np.array(img2)
        
            norm_img2 = normalizeData(img2)
        
            pca = PCA(n_components=2)
        
            data = pca.fit_transform(norm_img2)
        
            centroid = np.expand_dims(data.mean(axis=0), axis = 0)
        
            dist = [np.linalg.norm(data[i] - centroid) for i in range(len(data))]
        
            index_min = np.argmin(dist)
            indexes.append(index_min)
            
            class_rep = [self.classes[i][indexes[i]] for i in range(len(indexes))]
        
        return class_rep
        
    # returning final neg class and postive class path
    def return_item(self):
        
        # ORB similarity
        def orb_sim(img1, img2):
            
          orb = cv2.ORB_create()

          # detect keypoints and descriptors
          kp_a, desc_a = orb.detectAndCompute(img1, None)
          kp_b, desc_b = orb.detectAndCompute(img2, None)
          
          if desc_a is None or desc_b is None:
              return 0

          # define the bruteforce matcher object
          bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
          #perform matches. 
          matches = bf.match(desc_a, desc_b)
          #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
          similar_regions = [i for i in matches if i.distance < 50]  
          if len(matches) == 0:
            return 0
          return len(similar_regions) / len(matches)
        
        
        # neg rep index
        indx_neg = [self.rep_image(self.neg_images[i], self.neg_clusters[i]) \
                    for i in tqdm(range(len(self.neg_images)))] 
            
        # load pos img array    
        img_pos_array = []
        for i in range(len(self.indx)):
            img = cv2.imread(self.path + self.indx[i])
            img = cv2.resize(img, (self.img_size,self.img_size))
            img_pos_array.append(img)
        
        # load neg img array  
        img_neg_array = []
        for j in range(len(indx_neg)):
            img_neg_row = []
            for i in range(len(indx_neg[0])):
                img = cv2.imread(self.path + indx_neg[j][i])
                img = cv2.resize(img, (self.img_size,self.img_size))
                img_neg_row.append(img)
            img_neg_array.append(img_neg_row)
        
        # similarity between pos and neg images
        
        similarity = []
        for j in tqdm(range(len(img_pos_array))):
            row_similarity = []
            for i in range(self.cluster_num):
                similarity_value = orb_sim(img_pos_array[j], img_neg_array[j][i])
                row_similarity.append(similarity_value)
            similarity.append(row_similarity)
            
        # most similar index
        index_min = [np.argmin(similarity[i]) for i in range(len(similarity))]
        # selected final neg class
        final_neg_classes = [self.neg_clusters[i][e] for i, e in enumerate(index_min)]
        
        try:
            with open(self.save_path+"Caltech_neg_classes_%s_%s.pkl" %(self.mode, self.cluster_num), "wb") as fp:
                pickle.dump(final_neg_classes, fp)
    
            with open(self.save_path+"Caltech_pos_classes_%s_%s.pkl" %(self.mode, self.cluster_num), "wb") as fp:
                pickle.dump(self.classes, fp)
        except:
            pass
        
        end = time.time()
        print("The time of execution is :", round((end-self.start) / 60, 2), "minutes")
        
        return final_neg_classes, self.classes # selected neg cluster path, selected pos cluster path
    
root = '/mnt/d/OneDrive - Oklahoma A and M System/RA/Fall 23/BMAML/Code/miniImageNet/'
mode = 'test' 
         
final_neg_classes, final_pos_classes = BMAML(root, mode, cluster_num = 20).return_item() # takes approx. 2 hours to compute on RTX3080ti