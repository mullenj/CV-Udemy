# Object Detection using SSD
# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
#Define the detection function
def detect(image, ssd, transform):
    
    height, width = image.shape[:2]
    imageT = transform(image)[0]
    imageT = torch.from_numpy(imageT).permute(2,0,1)
    imageT = Variable(imageT.unsqueeze(0))
    x = ssd(imageT)
    detections = x.data
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j=0
        while detections[0,i,j,0] >= 0.6:
            pt = (detections[0,i,j,1:]*scale).numpy()
            cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,0,0), 2)
            cv2.putText(image, labelmap[i-1],(int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            j+=1
    return image
#generate SSD network
ssd = build_ssd('test')
ssd.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))
#create transformation
transform = BaseTransform(ssd.size, (104/256.0, 117/256.0, 123/256.0))
#open video, iterate on frames, and create new video with detected objects
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps=fps)
for i, image in enumerate(reader):
    image = detect(image, ssd.eval(), transform)
    writer.append_data(image)
    print(i)
writer.close()