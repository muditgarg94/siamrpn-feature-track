import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import math
from math import floor
refPt = []
cropping = False

# function for labelling object 
def click_and_crop(event, x, y, flags, param):

    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
 
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
         
        # draw a rectangle around the region of interest
        cv2.rectangle(img_copy, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", img_copy)

def feature_images(input_image) :

    '''function for computing the 49 feature images.
       Any output pixel=R*a+G*b+B*c and contains only linearly independent combinations
       a,b,c belongs to {-2,-1,0,1,2}.Also, {0,0,0} not allowed
       input : input image
       returns a list of all images'''

    image=input_image
    # Now, for all the feature spaces(49 in total..considering linear combinations of R,G,B values)

    feature_spaces=np.array([(1,1,1),(2,2,1),(1,1,0),(2,2,-1),(1,1,-1),(2,1,2),(2,1,1),(2,1,0),(2,1,-1),(2,1,-2),(1,0,1),(2,0,1),(1,0,0),
    (2,0,-1),(1,0,-1),(2,-1,2),(2,-1,1),(2,-1,0),(2,-1,-1),(2,-1,-2),(1,-1,1),(2,-2,1),(1,-1,0),(2,-2,-1),(1,-1,-1),(1,2,2),(1,2,1),(1,2,0),
    (1,2,-2),(1,2,-1),(1,1,2),(1,1,-2),(1,0,2),(1,0,-2),(1,-1,2),(1,-1,-2),(1,-2,2),(1,-2,1),(1,-2,0),(1,-2,-1),(1,-2,-2),(0,1,1)
    ,(0,2,1),(0,1,0),(0,2,-1),(0,1,-1),(0,1,2),(0,1,-2),(0,0,1)])
    max_possible=[]
    min_possible=[]
    # iterating for each feature space
    for i,feature in enumerate(feature_spaces) :
        maximum = max(0,feature[0])*255 + max(0,feature[1])*255 +max(0,feature[2])*255
        minimum = min(0,feature[0])*255 + min(0,feature[1])*255 +min(0,feature[2])*255
        max_possible.append(maximum)
        min_possible.append(minimum)

    h,w,c=image.shape
    new_image=np.zeros((h,w,c))
    image=image.astype('uint32')
    feature_images_list=[]
    #import pdb; pdb.set_trace()
    for i in range(0,49) :
        range_=max_possible[i]-min_possible[i]
        xyz=((image[:,:,0]*feature_spaces[i][0]+image[:,:,1]*feature_spaces[i][1]+image[:,:,2]*feature_spaces[i][2]-min_possible[i])*255)/range_
        new_image[:,:,0]= xyz
        new_image[:,:,1]= xyz
        new_image[:,:,2]= xyz
        #new_image[:,:,1]=((image[:,:,1]*feature_spaces[i][1]-min_possible[i])*255)/range_
        #new_image[:,:,2]=((image[:,:,2]*feature_spaces[i][2]-min_possible[i])*255)/range_
        new_image=new_image.astype('uint8')
        feature_images_list.append(new_image)

    return feature_images_list
#x = np.random.rand(1,2,3)
#y = feature_images(x)

def likelihood(img,obj_img,bg_img,h_,w_) :

    #img is the feature image
    '''input : feature image, labelled object, background images
       returns likelihood image and variance'''
    #global h_,w_   
    
    h,w,c=bg_img.shape
    bg_img[h_:h-h_,w_:w-w_,:] = 0  #segmenting surrounding from object pixels

    hist_obj = cv2.calcHist([obj_img],[0],None,[32],[0,256])
    hist_bg  = cv2.calcHist([bg_img],[0],None,[32],[0,256])
    hist_bg[0]=hist_bg[0]- sum(hist_obj) # for removing the effect of the dark pixels in the object region 

    #plotting object and background
    #plt.plot(hist_bg,color='b',label="background")
    #plt.plot(hist_obj,color='r',label="object")
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    #plt.show()

    # delta for avoiding divide by zero and zero in the logarithm
    delta=np.ones((32,1))*.0001

    #normalizing the histograms
    #p for object and q for object
    p=hist_obj/sum(hist_obj)
    q=hist_bg/sum(hist_bg)

    p=p.ravel()
    q=q.ravel()
    delta=delta.ravel()

    # defining the log likelihood

    temp1=np.maximum(p,delta)
    temp2=np.maximum(q,delta)
    L = np.log10(temp1)-np.log10(temp2)
    # compute variance
    VR_intra1 = variance(L,p)
    VR_intra2 = variance(L,q)
    VR_inter = variance(L,(p+q)/2)
    x=np.maximum(VR_intra1+VR_intra2,0.001)
    VR = VR_inter/(x)
    #print ("variance=",VR)
    #plotting likelihood
    #plt.figure()
    x=[0,35]
    y=[127,127]
    #plt.plot(x,y,color='black')
    #plt.plot(L)
    #plt.show()
    # for creating likelihood image
    L_=((L+4)*255)/8 
#    import pdb; pdb.set_trace()
#    print("hey", img)
    '''for i in range(32) :
        if L_[i]>127 :
            L_[i]=255
        elif L_[i]<127 :
            L_[i]=0
        else :
            L_[i]=127'''
    img=img//8
#    print(img)
    likelihood_img=L_[img]  # creating likelihood image

    #cv2.imshow("like",likelihood_img)
    #cv2.waitKey(0)

    return likelihood_img,VR

def variance(L,a) :

    # computes variance of a distribition with pdf=a
    temp1=np.sum(np.multiply(np.multiply(L,L),a))
    temp2=np.sum(np.multiply(a,L))
    temp2=temp2*temp2
    var=temp1-temp2
    #print temp1,temp2,"temp variance",a

    return var



def __init__(image,bbox):
    clone=image.copy()
    y1 = bbox[1]
    y2 = bbox[3] +y1
    x1 = bbox[0]
    x2 = bbox[2] +x1
    #cropped=img[floor(min_x):floor(min_x+bbox[2]),floor(min_y):floor(min_y+bbox[3])]
    roi = clone[floor(y1):floor(y2), floor(x1):floor(x2)]
    h,w,c=roi.shape
    h_=int(h*0.3)
    w_=int(w*0.3)
    bg_img_original= clone[floor(y1-h_):floor(y2+h_), floor(x1-w_):floor(x2+w_)]   # roi containing object and surroundings

    list_feature_images=feature_images(image)
    list_object_images =feature_images(roi)
    list_bg_images     =feature_images(bg_img_original)

    list_VR=[]
    list_likelihood_images=[]

    for i in range(49) :
        likelihood_image,VR = likelihood(list_feature_images[i],list_object_images[i],list_bg_images[i],h_,w_)
        list_VR.append(VR)
        list_likelihood_images.append(likelihood_image)
        
    sorted_VR=sorted(range(len(list_VR)),key=lambda x:list_VR[x],reverse=True)
    #print(list_VR)
    #print("Sorted Indices: ",sorted_VR)

    best_img=(list_likelihood_images[sorted_VR[0]]+list_likelihood_images[sorted_VR[0]]+list_likelihood_images[sorted_VR[0]])/3
    #cv2.imshow("image",best_img)

    return best_img
