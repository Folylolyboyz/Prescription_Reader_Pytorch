import os
from tqdm import tqdm
from urllib.request import urlopen

import torch
import torch.optim as optim

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from model import Network
from configs import ModelConfigs

torch.backends.cudnn.benchmark = True

datasetPath = "Dataset"
dataset = []
vocab = set()
maxLen = 0

words = open(os.path.join(datasetPath, "words.txt"), "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue

    lineSplit = line.split(" ")
    if lineSplit[1] == "err":
        continue

    folder1 = lineSplit[0][:3]
    folder2 = "-".join(lineSplit[0].split("-")[:2])
    fileName = lineSplit[0] + ".png"
    label = lineSplit[-1].rstrip("\n")

    relativePath = os.path.join(datasetPath, "words", folder1, folder2, fileName)
    if not os.path.exists(relativePath):
        print(f"File not found: {relativePath}")
        continue

    dataset.append([relativePath, label])
    vocab.update(list(label))
    maxLen = max(maxLen, len(label))

configs = ModelConfigs()

configs.vocab = "".join(sorted(vocab))
configs.max_text_length = maxLen
configs.save()

# Create a data provider for the dataset
dataProvider = DataProvider(
    dataset=dataset,
    workers = 20,
    skip_validation = True,
    use_multiprocessing = True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

trainDataProvider, validationDataProvider = dataProvider.split(split = 0.9)

trainDataProvider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), 
    ]

network = Network(len(configs.vocab), activation="leaky_relu", dropout=0.3)
loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    network = network.cuda()

earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.height, configs.width, 3), 
    verbose=1,
    metadata={"vocab": configs.vocab}
    )

model = Model(network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])
print(model.device)

model.fit(
    trainDataProvider, 
    validationDataProvider, 
    epochs=1000, 
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx]
    )

trainDataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
validationDataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))