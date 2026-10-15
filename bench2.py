from ultralytics import YOLO

human_model = YOLO('yolov8_human.pt')
human_model.export(format='engine', half=True)

flood_model = YOLO('yolov8_flood.pt', task='obb')
flood_model.export(format='engine', half=True)
