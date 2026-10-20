import cv2
import numpy as np 
import matplotlib.pyplot as plt
image_path = 'D:\Semester-5\DSIP\Lab_Manual\secondimage.jpg'
image = cv2.imread(image_path, 1)
# plt.imshow(image,cmap="gray")
# plt.show()  
# # print(image)
# print(image.shape)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image_rgb)
# plt.show()  


# r,g,b = cv2.split(image)
# fig, ax=plt.subplots(1,4,figsize=(16,4))
# ax[0].imshow(image)
# ax[0].set_title("Original")
# ax[1].imshow(r,cmap= 'Reds')
# ax[1].set_title("Red")
# ax[2].imshow(g,cmap= 'Greens')
# ax[2].set_title("Green")
# ax[3].imshow(b,cmap= 'Blues')
# ax[3].set_title("Blue")

# for a in ax : a.axis('on')
# plt.show()


# crop_image=image[117:310, 238:360]
# plt.imshow(crop_image)
# plt.show()


# resize_image=cv2.resize(image,(300,300))
# # plt.figure(figsize=(10,4))
# plt.imshow(resize_image)
# plt.show()


# weighted_kernel = np.asarray([[1,1,1],[1,2,1],[1,1,1]],dtype=np.float32)
# weighted_kernel = weighted_kernel /10 
# weighted_mean = cv2.filter2D(image_rgb , -1, weighted_kernel)
# plt.imshow(weighted_mean)
# plt.show()  


# weighted_kernel_5 = np.asarray([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=np.float32)
# weighted_kernel_5 = weighted_kernel_5 /25 
# weighted_mean_5 = cv2.filter2D(image_rgb , -1, weighted_kernel_5)
# plt.imshow(weighted_mean_5)
# plt.show()  


# g0= cv2.GaussianBlur(image_rgb ,(5,5),sigmaX=0)
# plt.imshow(g0)
# plt.show()  

# g1= cv2.GaussianBlur(image_rgb ,(5,5),sigmaX=1)
# plt.imshow(g1)
# plt.show()  
# g5= cv2.GaussianBlur(image_rgb ,(5,5),sigmaX=5)
# plt.imshow(g5)
# plt.show()  


laplacian  =cv2.Laplacian(image ,cv2.CV_64F)
laplacian  =cv2.Laplacian(image_rgb ,cv2.CV_64F)
# plt.imshow(laplacian)
# plt.show()  
# print(laplacian)

# laplacian_abs = cv2.convertScaleAbs(laplacian)
# plt.imshow(laplacian_abs)
# plt.show()  
# print(laplacian_abs)


# roberts_x= np.asarray([[1,0],[0,-1]],dtype=np.float32)
# roberts_y= np.array([[0,1],[-1,0]],dtype=np.float32)

# gx=cv2.filter2D(image_rgb.astype(np.float32),cv2.CV_32F , roberts_x)
# gy=cv2.filter2D(image_rgb.astype(np.float32),cv2.CV_32F , roberts_y)

# roberts  = cv2.magnitude(gx,gy)
# roberts = cv2.normalize(roberts , None ,0,255 , cv2.NORM_MINMAX).astype(np.uint8)

# plt.imshow(roberts)
# plt.show()  

prewitt_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype= np.float32)
prewitt_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype= np.float32)

px=cv2.filter2D(image_rgb.astype(np.float32),cv2.CV_32F , prewitt_x)
py=cv2.filter2D(image_rgb.astype(np.float32),cv2.CV_32F , prewitt_y)
prewitt = cv2.magnitude(px,py)

prewitt = cv2.normalize(prewitt , None , 0,255,cv2.NORM_MINMAX).astype(np.uint8)

plt.imshow(prewitt)
plt.show()
