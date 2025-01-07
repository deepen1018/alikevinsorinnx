import cv2
import numpy as np
from alike import ALike, configs
import time

def test_alike():
    # 初始化模型
    model = ALike(**configs['alike-t'], device='cuda')
    
    # 讀取測試圖像
    img = cv2.imread('/home/deepen/catkin_ws/src/ALIKE/assets/tum/1311868169.163498.png')
    if img is None:
        raise FileNotFoundError("Test image not found")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 運行檢測
    start = time.time()
    pred = model(img_rgb)
    end = time.time()
    
    # 打印結果
    print(f"Image shape: {img.shape}")
    print(f"Found {len(pred['keypoints'])} keypoints")
    print(f"Processing time: {(end-start):.3f}s")

if __name__ == "__main__":
    test_alike()
