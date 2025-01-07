import numpy as np
import alike
import torch
import cv2
import traceback
import time

class PyALikeDetector:
    def __init__(self):
        try:
            # GPU 可用性檢查
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            
            # 初始化 GPU 設置
            torch.cuda.set_device(0)
            self.device = torch.device('cuda:0')
            print(f"Using device: {self.device}")
            
            # 顯示初始 GPU 內存狀態
            print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"Initial CUDA memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
            
            # 初始化模型並移至 GPU
            with torch.cuda.device(self.device):
                self.model = alike.ALike(**alike.configs['alike-l'])
                self.model.cuda()
                print(f"Model initialized on: {next(self.model.parameters()).device}")
                
                # 進行測試推理
                try:
                    with torch.no_grad():
                        dummy_input = torch.rand(1, 3, 480, 752, device=self.device)
                        _ = self.model(dummy_input)
                        print("Test inference completed successfully")
                except Exception as e:
                    print(f"Test inference failed: {str(e)}")
                    raise
                
            print(f"After model load - Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            traceback.print_exc()
            raise

    
    # def detect(self, image, max_corners, quality_level, min_distance, mask=None):
    #     try:
    #         print("=== Python detect method start ===")
    #         print(f"Input image info: shape={image.shape}, dtype={image.dtype}")

    #         with torch.cuda.device(self.device):
    #             # 1. 圖像預處理
    #             if len(image.shape) == 2:
    #                 image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #             print(f"Processed image shape: {image.shape}")
                
    #             # 2. 轉換為PyTorch張量
    #             tensor = torch.from_numpy(image).float()
    #             tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    #             print(f"Tensor prepared: shape={tensor.shape}")
                
    #             # 3. 移至GPU並運行模型
    #             tensor = tensor.cuda(self.device, non_blocking=True)
    #             torch.cuda.synchronize()
                
    #             with torch.no_grad():
    #                 # 限制檢測點數
    #                 self.model.top_k = max_corners  # 確保不超過要求的最大點數
    #                 results = self.model(tensor, sort=True)  # 按分數排序
                
    #             print("Model inference completed")
                
    #             # 4. 處理結果
    #             if 'keypoints' in results:
    #                 keypoints = results['keypoints']
    #                 # 只保留前max_corners個點
    #                 if len(keypoints) > max_corners:
    #                     keypoints = keypoints[:max_corners]
    #                 return keypoints
                    
    #             return np.array([])
                
    #     except Exception as e:
    #         print(f"Detection error: {e}")
    #         traceback.print_exc()
    #         torch.cuda.empty_cache()
    #         return np.array([])
    def detect(self, image, max_corners, quality_level, min_distance, mask=None):
        try:
            start_time = time.time()
            with torch.cuda.device(self.device):
                # 预处理
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # mask处理
                if mask is not None:
                    binary_mask = (mask > 0).astype(np.uint8)
                    mask_tensor = torch.from_numpy(binary_mask).float().to(self.device)
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                
                # 图像处理
                tensor = torch.from_numpy(image).float()
                tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
                tensor = tensor.cuda(self.device, non_blocking=True)

                with torch.no_grad():
                    # 运行ALIKE获取特征点和分数
                    results = self.model(tensor)
                    scores_map = results.get('scores_map', None)
                    keypoints = results['keypoints']
                    scores = results.get('scores', None)
                    
                    # 转换scores为torch tensor，如果它不是
                    if scores is not None and not isinstance(scores, torch.Tensor):
                        scores = torch.tensor(scores, device=self.device)
                    if keypoints is not None and not isinstance(keypoints, torch.Tensor):
                        keypoints = torch.tensor(keypoints, device=self.device)
                    
                    # 应用mask
                    if mask is not None and scores_map is not None:
                        scores_map *= mask_tensor
                        valid_mask = torch.nn.functional.grid_sample(
                            mask_tensor, keypoints.view(1, 1, -1, 2),
                            mode='nearest', align_corners=True
                        )[0, 0, 0]
                        scores = scores * valid_mask

                    # 按分数排序
                    if scores is not None:
                        valid_mask = scores > 0
                        keypoints = keypoints[valid_mask]
                        scores = scores[valid_mask]
                        sorted_indices = torch.argsort(scores, dim=0, descending=True)
                        keypoints = keypoints[sorted_indices]

                    # 应用最小距离约束
                    final_keypoints = []
                    image_points = []
                    rows, cols = image.shape[:2]
                    
                    # 转换keypoints为numpy以便处理
                    keypoints_np = keypoints.cpu().numpy()
                    
                    for kpt in keypoints_np:
                        if len(final_keypoints) >= max_corners:
                            break
                        
                        # 转换到像素坐标
                        px = (kpt[0] + 1.0) * cols / 2.0
                        py = (kpt[1] + 1.0) * rows / 2.0
                        pixel_pt = np.array([px, py])
                        
                        # 检查最小距离
                        valid_point = True
                        for exist_pt in image_points:
                            if np.linalg.norm(pixel_pt - exist_pt) < min_distance:
                                valid_point = False
                                break
                        
                        if valid_point:
                            final_keypoints.append(kpt)
                            image_points.append(pixel_pt)
                    end_time = time.time()  # 添加這行
                    detection_time = end_time - start_time
                    #print(f"Detected {len(final_keypoints)} keypoints") # debug信息
                    print(f"Detection took {detection_time*1000:.2f}ms")#叫出時間
                    return np.array(final_keypoints) , detection_time

        except Exception as e:
            print(f"Detection error: {e}")
            traceback.print_exc()
            return np.array([]), 0.0 