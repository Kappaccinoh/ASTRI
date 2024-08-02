import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10

def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path

def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
    if input_type == "Image":
        return yolov10_inference(image, None, model_id, image_size, conf_threshold)
    else:
        return yolov10_inference(None, video, model_id, image_size, conf_threshold)

if __name__ == "__main__":
    image = "images/cars.jpg"
    video = ""
    model_id = "yolov10m"
    image_size = 1280
    conf_threshold = 0.5
    input_type = 'Image'

    run_inference(image, video, model_id, image_size, conf_threshold, input_type)
