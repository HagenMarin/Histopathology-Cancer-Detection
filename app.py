"""webapp serving the HCD Model"""
from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
import cv2

from architecture import ResNet18
from architecture import BasicBlock


app = Flask(__name__)

#Model loading
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
net = ResNet18(img_channels=3, num_layers=18, block=BasicBlock, num_classes=1)
net.load_state_dict(torch.load("model/ResNet18Holdout_2dropout.pickle"))

@app.route("/")
def home():
    """returns index.html when website is opened"""
    return render_template('index.html')

@app.route("/",methods=['POST'])
def predict():
    """returns a prediction when a image is uploaded"""
    transform = torch.nn.Sequential(
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(0.3)
    )
    transform.to(device)
    #net.eval()
    with torch.no_grad():
        image = request.files['file']
        image.save(f'user_uploads/{image.filename}')
        image = cv2.imread(f'user_uploads/{image.filename}')
        img = image.astype(float)/255
        img = img.reshape((1,3,96,96))
        batch_imgs = torch.tensor((1,3,96,96),dtype=torch.float32)
        batch_imgs=torch.tensor(img,dtype=torch.float32)
        batch_imgs.to(device)
        outputs = net(batch_imgs)
        for _ in range(4):
            outputs = torch.cat((outputs,net(transform(batch_imgs))),1)
        outputs = torch.mean(outputs,1,True)

    return jsonify({'prediction': outputs[0].item()})


if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
