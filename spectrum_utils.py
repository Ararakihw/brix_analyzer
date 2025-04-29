import cv2
import numpy as np
from scipy.interpolate import interp1d

TARGET_FEATURE_LENGTH = 1600

def extract_spectrum_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = np.mean(gray, axis=0)
    x_old = np.linspace(0, 1, len(feature))
    f = interp1d(x_old, feature, kind='linear')
    x_new = np.linspace(0, 1, TARGET_FEATURE_LENGTH)
    return f(x_new)