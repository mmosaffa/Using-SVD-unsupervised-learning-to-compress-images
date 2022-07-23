import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import PIL.Image
from PIL import Image
#Read both images 
im=Image.open('D:/Laptop backup 2/course work2/course works2/Data Science program/files/i1.jpg')
im2=Image.open('D:/Laptop backup 2/course work2/course works2/Data Science program/files/i2.jpg')

#Change the images to arrays
im=np.asarray(im)
im2=np.asarray(im2)
#'''
#Change to gray scale
grayim=im.dot([0.299, 0.587, 0.114])
grayim2=im2.dot([0.299, 0.587, 0.114])
#Perform SVD on a Healthy image
svd=np.linalg.svd(grayim)
U=svd[0]
D=svd[1]
V=svd[2].T

DD=np.zeros([42,57])
#Construct the 99 by 57 matrix of singular values
l=np.concatenate((np.diag(D),DD))

#Reconstruct the original image
imm=U.dot(l).dot(V.T)
imhat=np.zeros([99,57,3])
enu=((0,20),(1,10),(2,5))
'''
for (i,k) in enu:
 Uprime=U[:,0:k]
 Dprime=np.diag(D[0:k])
 Vprime=V[:,0:k]
 imhat[:,:,i]=Uprime.dot(Dprime).dot(Vprime.T)
'''
for (i,k) in enu:
 Uprime=U[:,-k:]
 Dprime=np.diag(D[-k:])
 Vprime=V[:,-k:]
#'''
#Reconstruct a compressed image
 

plt.subplot(221)
plt.imshow(grayim,cmap="gray")
plt.subplot(222)
plt.title("Estimated Error:{}\nK=20".format(np.round_(np.linalg.norm(imm-imhat[:,:,0]),2)),fontsize=8)
plt.imshow(imhat[:,:,0],cmap="gray")
plt.subplot(223)
plt.title('Estimated Error:{}\nK=10'.format(np.round_(np.linalg.norm(imm-imhat[:,:,1]),2)),fontsize=8)
plt.imshow(imhat[:,:,1],cmap="gray")
plt.subplot(224)
plt.title('Estimated Error: {}\nK=5'.format(np.round_(np.linalg.norm(imm-imhat[:,:,2]),2)),fontsize=8)

plt.imshow(imhat[:,:,2],cmap="gray")
plt.tight_layout()
#plt.show()
#'''
#Defective image

svd2=np.linalg.svd(grayim2)
U2=svd2[0]
D2=svd2[1]
V2=svd2[2].T
DD2=np.zeros([42,57])
l2=np.concatenate((np.diag(D2),DD2))
imm2=U2.dot(l2).dot(V2.T)
#'''
'''
UU=U2[np.arange(98,-1,-1),0]
plt.subplot(121)
plt.plot(UU,range(99),c='r')
plt.subplot(122)
plt.imshow(grayim2,cmap='gray')
'''
