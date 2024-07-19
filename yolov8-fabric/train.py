import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolov8-farbic/ultralytics/cfg/models/v8/yolov8-DCFE.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/root/autodl-fs/fabric/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                seed=42,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )