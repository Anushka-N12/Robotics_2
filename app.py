from flask import Flask, make_response, Response
from ultralytics import YOLO
import cam
import json, os
from dotenv import load_dotenv
load_dotenv('.env')

esp_cam = os.getenv('esp_cam')  

app = Flask(__name__)  
bin_model = YOLO("bin_best.pt")
trash_model = YOLO("trash_best.pt")

d = {}

@app.route('/', methods=['GET'])
def pred():
    # Take image
    print(cam.getimg(esp_cam))
    # Run through custom model 
    trash_results = trash_model('frame.jpg')
    # Filter by size
    objs = []
    for result in trash_results:
        result = result.cpu().boxes.numpy()
        if len(result.cls) > 0:  # If detected, with decent confidence
            # print(result)
            [x1,y1,x2,y2] = result.xyxy[0]
            og_shape = result.orig_shape

            # # Confirm with pre-trained model
            # passed = True
            # image = cv2.imread("frame.jpg")
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw a rectangle around obj
            # cropped_image = image[y1:y2, x1:x2]    # Crop the image to the rectangle
            # cv2.imwrite("cropped.jpg", cropped_image)    # Save in       
            # cr_results = pt_model("cropped.jpg")
            # for result in cr_results:
            #     result = result.cpu().boxes.numpy()
            #     if pt_classes[result.cls[0]] in ['person', 'bicycle', 'fire hydrant', 'stop sign', 'bench', 'bird', 'cat', 'dog', 'chair', 'potted plant', 'dining table']:
            #         passed = False

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

    # Run through custom model 
    bin_results = bin_model('frame.jpg')
    # Filter by size
    bin = (-1,-1)
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
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    app.run()
    # print(model.names)

# results = model(r'C:\Users\anush\Projects\Emirates_Robotics\venv\data\test\trash.mp4', show=True)
