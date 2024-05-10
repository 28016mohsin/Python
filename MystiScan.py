#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


import matplotlib.pyplot as plt 


# In[3]:


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'##contains the actual weights of the pre-trained neural network.


# In[4]:


model = cv2.dnn_DetectionModel(frozen_model,config_file)


# In[5]:


classLabels = []
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


# In[6]:


print(classLabels)


# In[7]:


model.setInputSize(320,320)
##Sets the scale factor for input images. 
##This scales pixel values to a range that the model expects, based on the model's training.
model.setInputScale(1.0/127.5) ## 255/2 = 127.5 
##Sets the mean value for each channel of the input data, 
##used for mean normalization which helps in converging during training
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


# In[8]:


#Read an image


img = cv2.imread('businessman-standing-by-car-and-private-jet-at-terminal-DRX24M.jpg')


# In[9]:


plt.imshow(img)


# In[10]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[11]:


classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)


# In[12]:


font = cv2.FONT_HERSHEY_PLAIN

for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd-1], (boxes[0] + 10, boxes[1] + 40), font, fontScale = 3, color=(0, 255, 0), thickness=3)


# In[13]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


cap = cv2.VideoCapture('1')

# Check if the video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    print(ClassIndex)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 8), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
                cv2.imshow('MystiScan', frame)

                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()


# In[136]:





# In[ ]:





# In[ ]:





# In[ ]:




