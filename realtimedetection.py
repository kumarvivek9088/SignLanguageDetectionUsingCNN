from keras.models import model_from_json
import cv2
import numpy as np

json_file = open("signlanguagedetectionmodel48x48.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel48x48.h5")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

cap = cv2.VideoCapture(0)
label = ['A', 'M', 'N', 'S', 'T', 'blank']
while True:
    _,frame = cap.read()
    cv2.rectangle(frame,(0,40),(300,300),(0, 165, 255),1)
    cropframe=frame[40:300,0:300]
    cropframe=cv2.cvtColor(cropframe,cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe,(48,48))
    cropframe = extract_features(cropframe)
    pred = model.predict(cropframe) 
    prediction_label = label[pred.argmax()]
    cv2.rectangle(frame, (0,0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred)*100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
    cv2.imshow("output",frame)
    cv2.waitKey(27)
    
cap.release()
cv2.destroyAllWindows()