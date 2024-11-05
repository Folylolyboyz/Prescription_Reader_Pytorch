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
        # print(self.metadata["vocab"])

        return text

def createWords(pageDir: str, imgHeight = 1000, kernelSize = 25, sigma = 11, theta = 7, minArea = 100):
    img = prepare_img(cv2.imread(pageDir), imgHeight)
    detections = detect(img, kernel_size=kernelSize, sigma=sigma, theta=theta, min_area=minArea)

    # sort detections: cluster into lines, then sort each line
    lines = sort_multiline(detections)

    # plot results
    plt.imshow(img, cmap='gray')
    num_colors = 7
    colors = plt.get_cmap('rainbow', num_colors)
    
    wordFolder = os.path.join("Inference", "Words")
    
    # Words Folder Cleanup
    if os.path.exists(wordFolder):
        for items in os.listdir(wordFolder):
            os.remove(os.path.join(wordFolder, items))
        os.rmdir(wordFolder)
    
    if os.path.exists(wordFolder) == False:
        os.mkdir(wordFolder)
    
    for line_idx, line in enumerate(lines):
        for word_idx, det in enumerate(line):
            xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
            ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
            plt.plot(xs, ys, c=colors(line_idx % num_colors))
            plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
            # print(det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.y)
            cropImg = img[det.bbox.y:det.bbox.y+det.bbox.h, det.bbox.x:det.bbox.x+det.bbox.w]
            fileName = f"line-{str(line_idx)}-word-{str(word_idx)}.jpg"
            filePath = os.path.join(wordFolder, fileName)
            cv2.imwrite(filePath, cropImg)
    
    wordPredict(wordFolder)
    plt.show()

def wordPredict(wordFolder = os.path.join("Inference", "Words")):
    imgFiles = os.listdir(wordFolder)

    model = ImageToWordModel(model_path="Models/202410180902/model.onnx")

    for imgName in imgFiles:
        imgDir = os.path.join(wordFolder, imgName)
        img = cv2.imread(imgDir)
        # print(img)
        pred = model.predict(img)
        print(f"{imgName} : {pred}")
    
    # Words Folder Cleanup After Prediction
    # if os.path.exists(wordFolder):
    #     for items in os.listdir(wordFolder):
    #     os.remove(os.path.join(wordFolder, items))
    # os.rmdir(wordFolder)
    
    

pageDir = os.path.join("Inference", "Page", "page1.png")
createWords(pageDir)