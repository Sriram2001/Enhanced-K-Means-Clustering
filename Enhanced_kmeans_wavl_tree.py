import numpy as np
import pandas as pd
from scipy.spatial import distance
import sys
import copy
import collections
import cv2
from PIL import Image
from scipy import misc
from wavl import*



class cluster:
    def __init__(self, clusterId, cluster_centroid):
        self.clusterId = clusterId
        self.clusterCentroid = cluster_centroid
        self.clusterList = []

    def removepoint(self, pointid):
        self.clusterList.remove(pointid)

    def addpoint(self, pointid):
        self.clusterList.append(pointid)

    def get_centroid(self, points):
        index = np.array(self.clusterList)
        selected_points = [points[i] for i in index]
        clusters = np.array(selected_points)
        self.cluster_centroid = np.mean(clusters, axis=0)

        return self.cluster_centroid

class Kmeans:
    def __init__(self, iterations, K):
        self.iterations = iterations
        self.K = K
        self.points = []
        self.clusters = []
        self.tree = WAVLTree()

    def Euclidian_distance(self, p1, center):
        return distance.euclidean(p1, center)

    def fit(self, all_points):
        self.points = all_points

    def run(self):
        if self.K>len(self.points):
            return
        self.points = np.array(self.points).astype(float)
        np.random.seed(42)
        random_index = np.random.choice(range(len(self.points)), self.K, replace=False)
        centroids = self.points[random_index]
        for i in range(self.K):
            self.clusters.append(cluster(i,centroids[i]))
        distance_to_centroid = np.zeros([len(self.points),self.K])
        for i in range(len(self.points)):
            k = 0
            min_dist = sys.maxsize
            for j in range(self.K):
                distance_to_centroid[i][j] = self.Euclidian_distance(self.points[i],centroids[j])

        for i in range(len(self.points)):
            self.tree.insert(i,distance_to_centroid[i])
            # self.tree[i] = distance_to_centroid[i]
            k = np.argmin(self.tree.find(i).value)
            self.clusters[k].addpoint(i)

        for i in range(self.K):
             print("Cluster List Initially",self.clusters[i].clusterList)
             print("Centroid Initially",centroids[i])
        changeArray = np.ones(self.K)
        for iter in range(self.iterations):

            #changed centroid count
            count = 0
            #old centroid (used for comparison later)
            oldcentroids = np.array(centroids)

            #Compute new centroids based on the points in the cluster
            for i in range(self.K):
                if(len(self.clusters[i].clusterList)==0):
                    centroids[i] = [0,0,0]
                else:
                    centroids[i] = self.clusters[i].get_centroid(self.points).astype(float)

            #checking what centroids have changed
            for j in range(self.K):
                if(collections.Counter(centroids[j]) == collections.Counter(oldcentroids[j])):
                    changeArray[j]=0
                    count = count + 1
                else:
                    changeArray[j]=1

            for i in range(self.K):
            #     print("Centroid new value",centroids[i])
                print("Changed array value",changeArray[i])

            #if none has changed then exit
            if(count==self.K):
                break

            #clear clusterList for values which have changed
            for i in range(self.K):
                self.clusters[i].clusterList.clear()

            #print("Initial distance to centroid",distance_to_centroid)

            #calculate distance of centroid
            for i in range(len(self.points)):
                for j in range(self.K):
                    if(changeArray[j]==1):
                        distance_to_centroid[i][j] = self.Euclidian_distance(self.points[i], centroids[j])

                pre = np.argmin(self.tree.find(i).value)
                self.tree.find(i).value = distance_to_centroid[i]
                k = np.argmin(self.tree.find(i).value)
                self.clusters[k].addpoint(i)


                # print("Tree values",self.tree[i])

            # print("Final distance to centroid", distance_to_centroid)
            #printing to check contents of a cluster
            for j in range(self.K):
                print("cluster contents", self.clusters[j].clusterList)

            #print cluster centroid
            for i in range(self.K):
                print("New cluster centroids",self.clusters[i].clusterCentroid)
            counter=0
            for i in range(self.K):
                counter = counter + len(self.clusters[i].clusterList)
            print("Final length of the values are",counter)

    def storeValues(self,replaceList):
        #save contents of the cluster to a file
        for i in range(self.K):
            for j in range(len(self.clusters[i].clusterList)):
                replaceList[self.clusters[i].clusterList[j]] = self.clusters[i].clusterCentroid.tolist()
        print(replaceList)
        return replaceList



image_seg_orig = cv2.imread("Shashi.jpg")
cap = cv2.resize(image_seg_orig, (256,256))
cap = cv2.cvtColor(cap,cv2.COLOR_BGR2RGB)
h,w,z = cap.shape
textFileList = []
biglist=[]
for y in range (0,h,1):
    list = []
    for x in range(0,w,1):
        color = cap[y,x]
        list.append(color)
        textFileList.append(color.tolist())
    biglist.append(list)
print(biglist)
print(textFileList)
array = np.array(biglist, dtype=np.uint8)
new_image = Image.fromarray(array)
new_image.save('compressed_orig.png')


k_means_text = open("image_pixelsnew.txt",'w')
for item in textFileList:
    k_means_text.write(str(item[0]))
    k_means_text.write(" ")
    k_means_text.write(str(item[1]))
    k_means_text.write(" ")
    k_means_text.write(str(item[2]))
    k_means_text.write('\n')
k_means_text.close()


obj = Kmeans(400,64)
data = np.loadtxt("image_pixelsnew.txt")
print(len(data),"is the length of the data")
obj.fit(data)
obj.run()
obj.storePointId()
valuesReplacedList = obj.storeValues(textFileList)

newImageList = []
k=0
for i in range(w):
    subList = []
    for j in range(h):
        subList.append(valuesReplacedList[k])
        k=k+1
    newImageList.append(subList)
print(newImageList)


array = np.array(newImageList, dtype=np.uint8)
new_image = Image.fromarray(array)
new_image.save('compressed_segmented_9.png')
