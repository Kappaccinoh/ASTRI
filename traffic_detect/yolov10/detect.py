import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10
from ultralytics.solutions import object_counter as oc

def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, verbose = False)
        # result_txt = results[0].verbose()
        summary = results[0].summary() 
        return summary
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


def count_specific_classes_in_image(image_path, model_id, classes_to_count):
    """Count specific classes of objects in an image."""
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    im0 = cv2.imread(image_path)
    assert im0 is not None, "Error reading image file"
    line_points = [(20, 400), (1080, 400)]
    counter = oc.ObjectCounter()
    counter.set_args(
        view_img=True,
        reg_pts=line_points,
        classes_names= model.names,
        draw_tracks=True
    )

    # Detect objects and count specific classes
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count, verbose=False)
    im0 = counter.start_counting(im0, tracks)
    # Save the modified image
    cv2.imwrite("output_image.jpg", im0)




if __name__ == "__main__":
    image = "Camera_Images/AID04218.jpg"
    video = ""
    model_id = "yolov10m"
    image_size = 1280
    conf_threshold = 0.5
    input_type = 'Image'

    run_inference(image, video, model_id, image_size, conf_threshold, input_type)
    
    # res = count_specific_classes_in_image(image, model_id, classes)

