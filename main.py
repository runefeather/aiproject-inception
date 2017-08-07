import sys
import inception
from inception import transfer_values_cache
import tumordata
import prettytensor as pt
import os
import tensorflow as tf
import numpy as np

from transfer_learning import main

num_classes = 2

def predict_avg(imgpath):
    s1, pred = main("split1", imgpath)
    # s2, pred2 = main("split2", imgpath)
    # s3, pred3 = main("split3", imgpath)
    # s4, pred4 = main("split4", imgpath)
    # s5, pred5 = main("split5", imgpath)

    print("Accuracy: ", s1, ", and prediction: ", pred)


if __name__ == '__main__':
    imgpath = []
    imgpath.append("D:\\AI stuff\\aiproject-inception-master\\breakhis\\BreaKHis_data\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\fibroadenoma\\SOB_B_F_14-23222AB\\40X\\SOB_B_F-14-23222AB-40-001.png")
    imgpath.append("D:\\AI stuff\\aiproject-inception-master\\breakhis\\BreaKHis_data\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\fibroadenoma\\SOB_B_F_14-23222AB\\40X\\SOB_B_F-14-23222AB-40-002.png")
    imgpath.append("D:\\AI stuff\\aiproject-inception-master\\breakhis\\BreaKHis_data\\BreaKHis_v1\\histology_slides\\breast\\benign\SOB\\fibroadenoma\\SOB_B_F_14-9133\\40X\\SOB_B_F-14-9133-40-021.png")
    imgpath.append("D:\\AI stuff\\aiproject-inception-master\\breakhis\\BreaKHis_data\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\tubular_adenoma\\SOB_B_TA_14-16184\\40X\\SOB_B_TA-14-16184-40-017.png")
    imgpath.append("D:\\AI stuff\\aiproject-inception-master\\breakhis\\BreaKHis_data\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\fibroadenoma\\SOB_B_F_14-9133\\40X\\SOB_B_F-14-9133-40-011.png")

    for i in imgpath:
        predict_avg(i)

