import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#'deprecation.showPyplotGlobalUse'


def obj_detection(my_img):
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    column1, column2 = st.columns(2)

    column1.subheader("Input image")
    st.text("")
    plt.figure(figsize = (16,16))
    plt.imshow(my_img)
    column1.pyplot(use_column_width=True)

    # YOLO model
    net = cv2.dnn.readNet("custom-yolov4-detector_best.weights","cfg/yolo4.cfg")

    #labels = []
    f= os.path.abspath('coco.names')
    #with open("coco.names") as f:
    labels = open(f).read().strip().split("\n")
    #labels = [line.strip() for line in f.readlines()]
    names_of_layer = net.getLayerNames()
    output_layers = [names_of_layer[i[0]-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(len(labels), 3))   


    # Image loading
    newImage = np.array(my_img.convert('RGB'))
    img = cv2.cvtColor(newImage,1)
    height,width,channels = img.shape


    # Objects detection (Converting into blobs)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   #(image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    classID = []
    confidences = []
    boxes =[]

    # SHOWING INFORMATION CONTAINED IN 'outputs' VARIABLE ON THE SCREEN
    for op in outputs:
        for detection in op:
            scores = detection[5:]
            class_id = np.argmax(scores)  
            confidence = scores[class_id] 
            if confidence > 0.5:   
                # OBJECT DETECTED
                #Get the coordinates of object: center,width,height  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  #width is the original width of image
                h = int(detection[3] * height) #height is the original height of the image

                # RECTANGLE COORDINATES
                x = int(center_x - w /2)   #Top-Left x
                y = int(center_y - h/2)   #Top-left y

                #To organize the objects in array so that we can extract them later
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                classID.append(class_id)

    score_threshold = st.sidebar.slider("Confidence_threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)      
    print(indexes)

   # font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            #To get the name of object
            label = str.upper((labels[classID[i]]))   
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)     
            items.append(label)


    st.text("")
    column2.subheader("Detected image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    column2.pyplot(use_column_width=True)

    if len(indexes)>0:
        st.success("Found {} Hairstyle - {}".format(len(indexes),[item for item in set(items)]))
    else:
        st.success("Found {} Hairstyle - {}".format(len(indexes),[item for item in set(items)]))


def main():
    
    st.title("African Hairstyles Detection")
    st.write("Description:This is a computer vision based app that detects  different African hairstyles to promote African culture and pride")

    #choice = st.radio("Choose an image of your choice")
    #st.write()

    #if choice == "Choose an image of your choice":
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    image_file = st.file_uploader("Upload", type=['jpg','png','jpeg'])

        #image_file is not None:
    my_img = Image.open(image_file)  
    obj_detection(my_img)

    #elif choice == "See an illustration":
        #my_img = Image.open("(9).jpeg")
        #obj_detection(my_img)

if __name__ == '__main__':
    main()

