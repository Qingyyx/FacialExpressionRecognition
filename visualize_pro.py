import numpy as np
import cv2
import os
from PIL import Image
import torch
from torchvision import transforms
from models import *
import json
from gradio import Interface, components as gr
import matplotlib.pyplot as plt

cut_size = 100
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(
        lambda crops: torch.
        stack([transforms.ToTensor()(crop) for crop in crops])
    ),
])

net = ResNet50()
checkpoint = torch.load(
    os.path.join(os.getcwd(), 'RAF_\\Test_model_epoch_32.t7'),
    map_location='cpu'
)
net.load_state_dict(checkpoint['net'])
net.eval()


def generate_bar_chart(class_names, scores):
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, scores)
    plt.xlabel('Class Names')
    plt.ylabel('Scores')
    plt.title('Scores Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图像文件
    image_path = "bar_chart.png"
    plt.savefig(image_path)
    plt.close()

    return image_path


def predict_expression(input_image):
    """
    接收用户上传的图像，处理后返回预测的表情类别和对应的概率条形图。
    """
    raw_img = np.array(input_image.copy())
    raw_img = cv2.resize(raw_img, (cut_size, cut_size))

    inputs = transform_test(Image.fromarray(raw_img))

    ncrops, c, h, w = inputs.shape
    inputs = inputs.view(-1, c, h, w)

    outputs = net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)
    # score = torch.softmax(outputs_avg, dim=0)
    score = torch.nn.functional.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)

    class_names = [
        'Angry', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness',
        'Surprise'
    ]

    expression = class_names[int(predicted)]
    bar_chart_image = generate_bar_chart(class_names, score.tolist())
    return expression, bar_chart_image


# input_image = cv2.imread('img.jpg')
# print(predict_expression(input_image))

# 使用Gradio界面
gradio_interface = Interface(
    fn=predict_expression,
    title="面部表情识别",
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Predicted Expression"),  # 展示预测的类别
        gr.Image(label="Probability Distribution"),  # 展示概率分布
    ],
    live=True
)

gradio_interface.launch(server_port=7860, share=True)
