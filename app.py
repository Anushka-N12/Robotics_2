from flask import Flask, make_response, Response
from ultralytics import YOLO
import cam
import json, os
from dotenv import load_dotenv
load_dotenv('.env')
import urllib
import cv2
import numpy as np
import ssl

url = 'https://10.10.148.14:8080'

# # while True:
#     imgResp = urllib3.urlopen(url)
#     imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
#     img = cv2.imdecode(imgNp, -1)
#     cv2.imwrite('temp',cv2.resize(img,(600,400)))

esp_cam = 'https://10.10.148.14:8080'

app = Flask(__name__)  
bin_model = YOLO("bin_best.pt")
trash_model = YOLO("trash_best.pt")

d = {}

@app.route('/', methods=['GET'])
def pred():
    # Take image
    cam_response = cam.getimg(esp_cam)
    print(cam_response)
    # Filter by size
    objs = []

        # Run through custom model 
    if cam_response:
        trash_results = trash_model('frame.jpg')
        for result in trash_results:
            result = result.cpu().boxes.numpy()
            if len(result.cls) > 0:  # If detected, with decent confidence
                # print(result)
                [x1,y1,x2,y2] = result.xyxy[0]
                og_shape = result.orig_shape

                # if passed == True and x2-x1 < og_shape[1]*0.75 and y2-y1 < og_shape[0]*0.75:  # Making sure obj is not too big
                if True:
                    objs.append(((x2+x1)//2, (y2+y1)//2))  # Adding center coords of object to list

    # Choose closest object
    if len(objs) > 0:    
        closest = (objs[0])
        # ideal = (og_shape[1]//2, og_shape[0]*0.75)
        for i in objs[1:]:
            if i[1] < closest[1]:
                closest = i
        print(closest)
    else:
        closest = (-1, -1)

    # Send result
    # Create a JSON response and set custom header
    d['trash'] = closest

    # Filter by size
    bin = (-1,-1)

    # Run through custom model 
    if cam_response:
        bin_results = bin_model('frame.jpg')
        result = bin_results[0].cpu().boxes.numpy()
        if len(result.cls) > 0:  # If detected, with decent confidence
            # print(result)
            [x1,y1,x2,y2] = result.xyxy[0]
            og_shape = result.orig_shape
            if True:
                bin = (((x2+x1)//2, (y2+y1)//2))  # Adding center coords of object to list
    # Send result
    # Create a JSON response and set custom header
    d['bin'] = bin

    response = json.dumps([d])
    response = Response(response, status=200, content_type='application/json')
    response.headers['X-My-Header'] = 'foo'
    return response, 200


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
    # print(model.names)

# results = model(r'C:\Users\anush\Projects\Emirates_Robotics\venv\data\test\trash.mp4', show=True)
