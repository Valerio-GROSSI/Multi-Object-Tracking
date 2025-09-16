from pathlib import Path
import argparse
import shutil
from datetime import datetime
import torch
from ultralytics import YOLO
from Reid_models import DEEPSORT_REID_MODELS, REID_MODEL_WEIGHTS_URLS
from Tracker_models import create_tracker, extract_embeddings, tracker_update
import torchreid
import cv2
import urllib.request
import numpy as np
import warnings
import configparser
import subprocess

## CLI Arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--input', type=str, default='MOT17',
                    help='Input directory name')
parser.add_argument('--yolo_model', nargs='+', type=str, default=['yolov8n.pt', 'yolo11m.pt'],
                    help='YOLO model name')
parser.add_argument('--reid_model', nargs='+', type=str, default=['osnet_x0_25_market1501.pt', 'mobilenetv2_x1_0_market1501.pt'],
                    help='ReID model name')
parser.add_argument('--tracker', nargs='+', type=str, default=['deepsort', 'bytetrack', 'botsort'],
                    help='Tracker name')
parser.add_argument('--gen_det_images', action='store_true',
                    help='Generate detection images')
parser.add_argument('--gen_track_images', action='store_true',
                    help='Generate tracking images')
parser.add_argument('--output', type=Path, default='runs',
                    help='Output parent directory name')
parser.add_argument('--from_detections', action='store_true',
                    help='Authorize tracking from input detections folders if exist')
args = parser.parse_args()

## Download TrackEval
if not Path("TrackEval").exists():
    subprocess.run(["git", "clone", "https://github.com/JonathonLuiten/TrackEval.git"])
    subprocess.run(["pip", "install", "-e", "TrackEval"])
else:
    print("TrackEval already exists, skipping clone and install.")

## Definition of paths
ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "Inputs"

# In this project, we use the 'mot_challenge' format
# The format defines the arrangement, structure and expected content of the data
# Other formats exist such as "coco", "kitti", etc. and are not handled here
TRACKEVAL_INPUTS = ROOT / "TrackEval" / "data"
TRACKEVAL_INPUTS_GT = TRACKEVAL_INPUTS / "gt" / "mot_challenge"
TRACKEVAL_INPUTS_TRACKERS = TRACKEVAL_INPUTS / "trackers" / "mot_challenge"

(TRACKEVAL_INPUTS_GT / "seqmaps").mkdir(parents=True, exist_ok=True)
TRACKEVAL_INPUTS_TRACKERS.mkdir(parents=True, exist_ok=True)

input_folder = INPUTS / Path(args.input)

## Copy of inputs to the TrackEval folder
if (input_folder / "train").exists():
    input_folder_train = input_folder / "train"  # Contains sequences with ground truth information
    path_dirs_train = [dir.resolve() for dir in input_folder_train.iterdir() if dir.is_dir()]  # list of sequences paths
    seqmap_file_train = (TRACKEVAL_INPUTS_GT / "seqmaps" / f"{args.input}-train.txt")
    with open(seqmap_file_train, 'w') as f:
        for path in path_dirs_train:
            dest_dir = TRACKEVAL_INPUTS_GT / f"{args.input}-train" / path.name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(path, dest_dir)
            if (dest_dir / "img1").exists():
                shutil.rmtree(dest_dir / "img1")
            if (dest_dir / "det").exists() and args.from_detections is False:
                shutil.rmtree(dest_dir / "det")
            f.write(f"{path.name}\n")  # Consideration of all the sequences in input_folder / "train"
if (input_folder / "test").exists():
    input_folder_test = input_folder / "test"  # Contains sequences without ground truth information
    path_dirs_test = [dir.resolve() for dir in input_folder_test.iterdir() if dir.is_dir()]  # list of sequences paths
    seqmap_file_test = (TRACKEVAL_INPUTS_GT / "seqmaps" / f"{args.input}-test.txt")
    with open(seqmap_file_test, 'w') as f:
        for path in path_dirs_test:
            dest_dir = TRACKEVAL_INPUTS_GT / f"{args.input}-test" / path.name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(path, dest_dir)
            if (dest_dir / "img1").exists():
                shutil.rmtree(dest_dir / "img1")
            if (dest_dir / "det").exists() and args.from_detections is False:
                shutil.rmtree(dest_dir / "det")
            f.write(f"{path.name}\n")  # Consideration of all the sequences in input_folder / "test"
if not (input_folder / "train").exists() and not (input_folder / "test").exists():
    raise ValueError(f"Input folder {input_folder} does not contain 'train' or 'test' subfolders.")

train_image_paths = [(path / "img1", "train") for path in path_dirs_train] if 'path_dirs_train' in locals() else []
test_image_paths = [(path / "img1", "test") for path in path_dirs_test] if 'path_dirs_test' in locals() else []
all_image_paths = test_image_paths + train_image_paths

## Preparing models
OUTPUTS = ROOT / "Outputs"
DETECTION_MODELS = ROOT / "Detection_models"
EMBEDDING_MODELS = ROOT / "Embedding_models"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = OUTPUTS / Path(args.output) / f"{args.input}_{timestamp}"

args.yolo_model = [DETECTION_MODELS / Path(m) for m in args.yolo_model]
args.reid_model = [EMBEDDING_MODELS / Path(m) for m in args.reid_model]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading all YOLO models via ultralytics
yolo_models = [YOLO(model_path) for model_path in args.yolo_model]
if args.from_detections:
    yolo_models.insert(0, "det")

# Loading all ReID model weights, and ReID models via torchreid if deepsort is specified in CLI
reid_models = {}
for reid_model_weights_path in args.reid_model:
    reid_model_weights_name = reid_model_weights_path.name

    # Loading ReID model weights
    if not reid_model_weights_path.exists():
        reid_model_weights_url = REID_MODEL_WEIGHTS_URLS.get(reid_model_weights_name)
        if reid_model_weights_url is None:
            raise ValueError(f"No URL found for ReID model weights: {reid_model_weights_name}, please check the WEIGHTS_URLS dictionary.")
        else:
            reid_model_weights_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(reid_model_weights_url, reid_model_weights_path)

    # Loading ReID models via torchreid if they exist, for deepsort only
    reid_model = None
    if 'deepsort' in args.tracker:
        reid_model_name = None
        for name in DEEPSORT_REID_MODELS:
            if name in reid_model_weights_name:
                reid_model_name = name
                break

        if reid_model_name is not None:
            reid_model_class = getattr(torchreid.models, reid_model_name)
            reid_model = reid_model_class(num_classes=1000, loss='softmax')
            torchreid.utils.load_pretrained_weights(reid_model, reid_model_weights_path)

            if hasattr(reid_model, 'classifier'):
                reid_model.classifier = torch.nn.Identity()
            elif hasattr(reid_model, 'fc'):
                reid_model.fc = torch.nn.Identity()

            reid_model = reid_model.to(device).eval()

    reid_models[reid_model_weights_path] = reid_model

## Pipeline
for image_path, split in all_image_paths:  # For each sequence
    image_path_name = image_path.parent.name  # Name of the sequence

    for yolo_model in yolo_models:  # For each YOLO model
        if yolo_model == "det": # default detection model
            yolo_name = "det"

            # Copy of input default detection file for the sequence to output folder
            det_dir = output_dir / image_path_name / "det" / "detections"
            det_dir.mkdir(parents=True, exist_ok=True)
            det_path = det_dir / "det.txt"
            if (image_path.parent / "det" / "det.txt").exists():
                shutil.copyfile(image_path.parent / "det" / "det.txt", det_path)
            else:
                raise FileNotFoundError(f"Detection file not found: {image_path.parent / 'det' / 'det.txt'}")

            # Creation of list for storing sequence features for future tracking
            all_detections = []
        
            sparse_all_frame_id = []
            sparse_all_boxes = []
            sparse_all_confs = []
            sparse_all_class_ids = []
            with det_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    liste = [elm.strip() for elm in line.split(",")]
                    frame_id = int(float(liste[0]))
                    x, y, w, h = map(float, liste[2:6])
                    conf = float(liste[6])
                    class_id = 0

                    sparse_all_frame_id.append(frame_id)
                    sparse_all_boxes.append((x, y, x + w, y + h))
                    sparse_all_confs.append(conf)
                    sparse_all_class_ids.append(class_id)

            idx = np.argsort(sparse_all_frame_id)
            sparse_all_frame_id = np.array(sparse_all_frame_id, dtype=int)[idx]
            sparse_all_boxes = np.array(sparse_all_boxes, dtype=float)[idx]
            sparse_all_confs = np.array(sparse_all_confs, dtype=float)[idx]
            sparse_all_class_ids = np.array(sparse_all_class_ids, dtype=int)[idx]

            split_points = np.where(np.diff(sparse_all_frame_id)!=0)[0] + 1
            sparse_all_frame_id = np.split(sparse_all_frame_id, split_points)
            sparse_all_boxes = np.split(sparse_all_boxes, split_points)
            sparse_all_confs = np.split(sparse_all_confs, split_points)
            sparse_all_class_ids = np.split(sparse_all_class_ids, split_points)

            EMPTY_BOXES = np.empty((0,4), dtype=np.float32)
            EMPTY_CONFS = np.empty((0,), dtype=np.float32)
            EMPTY_CLASS_IDS = np.empty((0,), dtype=int)

            ini_path = next((image_path.parent.glob("*.ini")), None)
            if ini_path is None:
                if image_path.exists():
                    num_frames = range(1, len(image_path.iterdir()) + 1)
                    warnings.warn(f"Attention, the number of frames for the sequence {image_path.parent} "
                                f"has been determined from the number of elements contained in {image_path.name}. "
                                f"It is rather recommended to provide the conventional .ini file in {image_path.parent}.")
                else:
                    raise ValueError(f"No way to determine the number of frames for the sequence "
                                    f"{image_path.parent} of {split}. Please provide in {image_path.parent} "
                                    f"the conventional .ini file or at a minimum a /img1 directory of all images "
                                    f"used for generating the detection file.")
            cfg = configparser.ConfigParser()
            cfg.read(ini_path)
            if "Sequence" not in cfg:
                raise KeyError(f"Section [Sequence] missing from the .ini file {ini_path}")
            seq = cfg["Sequence"]
            if "seqLength" not in seq:
                raise KeyError(f"Key [seqLength] missing from the section [Sequence] of the .ini file {ini_path}")
            num_frames = range(1, int(float(seq["seqLength"])) + 1)

            j=0
            all_frame_id, all_boxes, all_confs, all_class_ids = [], [], [], []
            for n in num_frames:
                if j < len(sparse_all_frame_id) and int(sparse_all_frame_id[j][0]) == n:
                    all_frame_id.append(sparse_all_frame_id[j])
                    all_boxes.append(sparse_all_boxes[j])
                    all_confs.append(sparse_all_confs[j])
                    all_class_ids.append(sparse_all_class_ids[j])
                    j += 1
                else:
                    all_frame_id.append(n)
                    all_boxes.append(EMPTY_BOXES)
                    all_confs.append(EMPTY_CONFS)
                    all_class_ids.append(EMPTY_CLASS_IDS)

            all_frame_id = [x[0] if isinstance(x, (np.ndarray)) else x for x in all_frame_id] # =num_frames =[1,2,..Nframes]
            all_img = [cv2.imread(str(img_path)) for img_path in sorted(image_path.iterdir())]
            all_img_name = [img_path.stem for img_path in sorted(image_path.iterdir())]

            # Storage of image features in the current sequence for future tracking
            all_detections = list(zip(all_img, all_boxes, all_confs, all_class_ids, all_frame_id, all_img_name))

            # Creation of detections output image directory for the sequence from the input default detection file
            if args.gen_det_images:
                det_img_dir = output_dir / image_path_name / "det" / "img1"
                det_img_dir.mkdir(parents=True, exist_ok=True)

                i = 0
                num_frame = 0
                for img_path in sorted(image_path.iterdir()):
                    img = cv2.imread(str(img_path))
                    img_name = img_path.stem
                    num_frame += 1

                    if img is None:
                        raise FileNotFoundError(f"Impossible de charger l'image : {img_path}. ")

                    out_img = img.copy()

                    if i < len(sparse_all_frame_id) and int(sparse_all_frame_id[i][0]) == num_frame:  # If the current frame has detections
                        for j, (x1, y1, x2, y2) in enumerate(sparse_all_boxes[i]):
                            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                            cv2.rectangle(out_img, p1, p2, (0, 255, 0), 2)
                            cv2.putText(
                                out_img,
                                f"{sparse_all_confs[i][j]:.2f}",
                                (p1[0], max(0, p1[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA
                            )
                        i += 1

                    cv2.imwrite(str(det_img_dir / f"{img_name}.jpg"), out_img)

        else:
            yolo_name = Path(yolo_model.model_name).stem

            # Creation of detections output images directory for the sequence
            if args.gen_det_images:
                det_img_dir = output_dir / image_path_name / yolo_name / "img1"
                det_img_dir.mkdir(parents=True, exist_ok=True)

            # Creation of detection file for the sequence
            detection_file_glb = TRACKEVAL_INPUTS_TRACKERS / f"{args.input}-{split}" / f"{yolo_name}_{timestamp}" / "data" / f"{image_path_name}.txt"
            detection_file_glb.parent.mkdir(parents=True, exist_ok=True)
            detects_f = open(detection_file_glb, "w")

            # Creation of lists for storing boxes and other sequence features for future tracking
            all_detections = []
            all_boxes = []
            frame_id = 1

            for img_path in sorted(image_path.iterdir()):  # For each image in the sequence
                img = cv2.imread(img_path)
                img_name = img_path.stem
                if img is None:
                    raise FileNotFoundError(f"Impossible de charger l'image : {img_path}. ")

                results = yolo_model(img, device=device, classes=[0])[0]  # Execution model YOLO / detection of person class only

                boxes = results.boxes.xyxy.cpu().numpy()  # Coordinates of the boxes (x1, y1, x2, y2)
                confs = results.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = results.boxes.cls.cpu().numpy().astype(int)

                # Creation of output image with detections
                if args.gen_det_images:
                    out_img = img.copy()
                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                        cv2.rectangle(out_img, p1, p2, (0, 255, 0), 2)
                        cv2.putText(
                            out_img,
                            f"{confs[i]:.2f}",
                            (p1[0], max(0, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA
                        )
                    cv2.imwrite(str(det_img_dir / f"{img_name}.jpg"), out_img)

                # Creation of detection file for the current image
                detection_file = output_dir / image_path_name / yolo_name / "detections" / f"{img_name}.txt"
                detection_file.parent.mkdir(parents=True, exist_ok=True)

                with detection_file.open("w") as f:
                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        w = x2 - x1
                        h = y2 - y1
                        score = confs[i]
                        cls = class_ids[i]
                        track_id = -1  # no tracking only detection
                        f.write(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.6f},-1,-1,-1\n")
                        detects_f.write(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.6f},-1,-1,-1\n")

                # Storage of boxes and other image features in the current sequence for future tracking
                all_detections.append((img, boxes, confs, class_ids, frame_id, img_name))
                all_boxes.extend([[frame_id, *b] for b in boxes])
                frame_id += 1

            detects_f.close()

            # Creation of only boxes file for the sequence
            if all_boxes:
                (output_dir / image_path_name / yolo_name / "boxes").mkdir(parents=True, exist_ok=True)
                np.savetxt(output_dir / image_path_name / yolo_name / "boxes" / "boxes.txt", np.array(all_boxes), fmt="%.2f", delimiter=",")

        for tracker_name in args.tracker:  # For each tracker
            if tracker_name == 'bytetrack':
                # Loading Tracker
                tracker = create_tracker(tracker_name)

                # Creation of tracking output images directory for the sequence
                if args.gen_track_images:
                    track_img_dir = output_dir / image_path_name / yolo_name / tracker_name / "img1"
                    track_img_dir.mkdir(parents=True, exist_ok=True)

                # Creation of tracking file (only Tracking no ReID) for the sequence
                tracking_file_glb = TRACKEVAL_INPUTS_TRACKERS / f"{args.input}-{split}" / f"{yolo_name}_{tracker_name}_{timestamp}" / "data" / f"{image_path_name}.txt"
                tracking_file_glb.parent.mkdir(parents=True, exist_ok=True)
                tracks_f = open(tracking_file_glb, "w")

                # Frame detections by frame detections processing
                for img, boxes, confs, class_ids, frame_id, img_name in all_detections:
                    embs = None
                    tracks = tracker_update(tracker, tracker_name, boxes, confs, class_ids, img, embs)  # Tracking data for the current frame

                    # Creation of output image with tracking
                    if args.gen_track_images:
                        out_img = img.copy()
                        for tid, x1, y1, x2, y2 in tracks:
                            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                            cv2.rectangle(out_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(out_img, f"ID {tid}", (x1, max(0, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.imwrite(str(track_img_dir / f"{img_name}.jpg"), out_img)

                    # Creation of tracking file for the current image
                    tracking_file = output_dir / image_path_name / yolo_name / tracker_name / "tracking" / f"{img_name}.txt"
                    tracking_file.parent.mkdir(parents=True, exist_ok=True)

                    with tracking_file.open("w") as f:
                        for tid, x1, y1, x2, y2 in tracks:
                            w, h = (x2 - x1), (y2 - y1)
                            f.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
                            tracks_f.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

                tracks_f.close()

            else:
                for reid_model_weights_path, reid_model in reid_models.items():  # For each ReID model
                    reid_model_weights_name = reid_model_weights_path.stem

                    if tracker_name != 'deepsort' or tracker_name == 'deepsort' and reid_model is not None:

                        # Loading Tracker
                        tracker = create_tracker(tracker_name, reid_weights=reid_model_weights_path)
                        
                        # Creation of tracking output images directory for the sequence
                        if args.gen_track_images:
                            track_img_dir = output_dir / image_path_name / yolo_name / tracker_name / reid_model_weights_name / "img1"
                            track_img_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Creation of tracking file (combination ReID + Tracking) for the sequence
                        tracking_file_glb = TRACKEVAL_INPUTS_TRACKERS / f"{args.input}-{split}" / f"{yolo_name}_{reid_model_weights_name}_{tracker_name}_{timestamp}" / "data" / f"{image_path_name}.txt"
                        tracking_file_glb.parent.mkdir(parents=True, exist_ok=True)
                        tracks_f = open(tracking_file_glb, "w")

                        # Creation of a list for storing embeddings (for DeepSORT)
                        all_embeddings = []

                        # Frame detections by frame detections processing
                        for img, boxes, confs, class_ids, frame_id, img_name in all_detections:
                            embs = None
                            if tracker_name == 'deepsort':
                                embs = extract_embeddings(img, boxes, reid_model, device)  # Extract embeddings for the current frame

                            tracks = tracker_update(tracker, tracker_name, boxes, confs, class_ids, img, embs)  # Tracking data for the current frame

                            # Creation of output image with tracking
                            if args.gen_track_images:
                                out_img = img.copy()
                                for tid, x1, y1, x2, y2 in tracks:
                                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                                    cv2.rectangle(out_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                    cv2.putText(out_img, f"ID {tid}", (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                                cv2.imwrite(str(track_img_dir / f"{img_name}.jpg"), out_img)
                            
                            # Creation of tracking file for the current image
                            tracking_file = output_dir / image_path_name / yolo_name / tracker_name / reid_model_weights_name / "tracking" / f"{img_name}.txt"
                            tracking_file.parent.mkdir(parents=True, exist_ok=True)

                            with tracking_file.open("w") as f:
                                for tid, x1, y1, x2, y2 in tracks:
                                    w, h = (x2 - x1), (y2 - y1)
                                    f.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
                                    tracks_f.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

                            # Storage of embeddings for the current sequence
                            if embs is not None and len(embs) > 0:
                                all_embeddings.append(embs)

                        tracks_f.close()

                        # Storage of embeddings for the current sequence if DeepSORT is used with valid ReID model
                        if tracker_name == 'deepsort' and all_embeddings:
                            all_embeddings_array = np.vstack(all_embeddings)
                            (output_dir / image_path_name / yolo_name / tracker_name / reid_model_weights_name / "embeddings").mkdir(parents=True, exist_ok=True)
                            np.save(output_dir / image_path_name / yolo_name / tracker_name / reid_model_weights_name / "embeddings" / "embeddings.npy", all_embeddings_array)
