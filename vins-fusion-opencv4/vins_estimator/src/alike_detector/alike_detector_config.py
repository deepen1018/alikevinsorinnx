# src/alike_detector/alike_detector_config.py
import os

# 獲取當前文件的目錄
DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DETECTOR_DIR, 'models')

# 確保模型目錄存在
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)