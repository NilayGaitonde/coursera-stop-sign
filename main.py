import torch
import cv2

def load_yolo():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def check_objs(model,image,n_classes=5):
    results = model(image)
    return results.xyxy[0][:n_classes]



def draw_boxes(model, frame, results,yoga_pose=None,score=None):
    for obj in results:
        x1,y1,x2,y2,confidence,label=obj
        label = model.names[int(label)]
        if y1<20:
            y1=20
        if y2>710:
            y2=700
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        # print(x1,y1,x2,y2,confidence,label)
        cv2.putText(frame,f'{label} ({confidence:.2f}) is doing the {yoga_pose} pose ({score})',(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    return frame
        
if __name__=='__main__':
    model = load_yolo()
    load_img = "stop/14.jpg"
    img = cv2.imread(load_img)
    results = model(load_img)
    # results.show()
    labels = list()
    for obj in results.xyxy[0]:
        confidence = obj[4].item() * 100
        label = model.names[int(obj[5])]
        labels.append(label)
    if 'stop sign' in labels:
        cv2.putText(img,'Stop sign detected',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    else:
        cv2.putText(img,'Stop sign not detected',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imwrite('outputs/output11.jpg', img)
    