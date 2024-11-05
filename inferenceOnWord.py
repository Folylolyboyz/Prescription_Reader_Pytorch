import cv2
import numpy as np
import os

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder

from wordDetector import detect, prepare_img, sort_multiline
from matplotlib import pyplot as plt

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.metadata["vocab"])[0]

        return text

def wordPredict(imgDir):
    model = ImageToWordModel(model_path="Models/08_handwriting_recognition_torch/202410180902/model.onnx")
    
    img = cv2.imread(imgDir)
    pred = model.predict(img)
    print(f"{pred}")


imgDir = os.path.join("Dataset", "words","b01" ,"b01-004","b01-004-00-02.png")
# imgDir = os.path.join("Inference", "Words", "line-0-word-0.jpg")
wordPredict(imgDir)