import hydra
import logging

import numpy as np
import torch

from src.Data.dataset import *
from src.Data.dataVisualize import *
from src.Data.dataAugment import *
from src.Model.EfficientNetB0 import *
# from src.Model.ResNet50x import *
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from src.train import *
from torchsummary import summary
from src.FaceDetection import *
from src.Filter import *
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
@hydra.main(version_base= None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # logger.info("Loading data")
    # data_train = IDataset(cfg.dataset.data_dir + "\\" + cfg.dataset.train_data.xml_file)
    # data_test = IDataset(cfg.dataset.data_dir +"\\"+ cfg.dataset.test_data.xml_file)
    # logger.info("Load success")
    #
    # logger.info("Augmentation Data")
    # A = DataAugment()
    # # train data
    # images_train, landmarks_train = data_train.images, data_train.landmarks
    # images_train_transform, landmarks_train_transform = A.norm_transform_dataset(images_train, landmarks_train,
    #                                                                            A.transform_train)
    # train_set_images, train_set_landmarks = images_train_transform, landmarks_train_transform
    #
    # # test data
    # images_test, landmarks_test = data_test.images, data_test.landmarks
    # images_test_transform, landmarks_test_transform = A.norm_transform_dataset(images_test, landmarks_test, A.transform_test)
    # test_set_images, test_set_landmarks = images_test_transform, landmarks_test_transform
    # logger.info("Augmentation Data successful")
    #
    # logger.info("Create Data for model")
    # # train data
    # data_train = LandmarkData(train_set_images, train_set_landmarks)
    # train_set = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=4)
    # # test data
    # data_test = LandmarkData(test_set_images, test_set_landmarks)
    # test_set = DataLoader(data_test, batch_size=32, shuffle=False, num_workers=4)
    # logger.info("Create Data for model successful")

    model = EfficientNetB0(68)

    # for param in model.model.parameters():
    #     param.requires_grad = False
    #
    # for param in model.model.classifier.parameters():
    #     param.requires_grad = True
    #
    # # Unfreeze model
    # for param in model.model.parameters():
    #     param.requires_grad = True
    # print(model)
    # summary(model, input_size=(3, 224, 224))
    # model.load_state_dict(torch.load('best_model_2.pth'))
    # model = train_model(model, train_set, test_set, 2)

    # A = ResNet50X()
    # print(A.model.summary())

if __name__ == '__main__':
    main()

