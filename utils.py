import os
import time
from sys import stderr


splits_LOSO = ['data_1.csv', 'data_2.csv', 'data_3.csv', 'data_4.csv', 'data_5.csv']
splits_LOUO = ['data_B.csv', 'data_C.csv', 'data_D.csv', 'data_E.csv', 'data_F.csv', 'data_G.csv', 'data_H.csv', 'data_I.csv']

splits_LOUO_NP = ['data_B.csv', 'data_C.csv', 'data_D.csv', 'data_E.csv', 'data_F.csv', 'data_H.csv', 'data_I.csv']

gestures_SU = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
gestures_NP = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G11']
gestures_KT = ['G1', 'G11', 'G12', 'G13', 'G14', 'G15']


def log(file, msg):
    """Log a message.

    :param file: File object to which the message will be written.
    :param msg:  Message to log (str).
    """
    print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
    file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
