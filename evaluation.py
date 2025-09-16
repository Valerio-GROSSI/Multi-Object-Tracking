import sys, runpy
from pathlib import Path
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent
TRACKEVAL_INPUTS = ROOT / "TrackEval" / "data"
TRACKEVAL_INPUTS_GT = TRACKEVAL_INPUTS / "gt" / "mot_challenge"
TRACKEVAL_INPUTS_TRACKERS = TRACKEVAL_INPUTS / "trackers" / "mot_challenge"

# list of detection file paths that one whishes to evaluate via TrackEval
detection_file_glb_list = [
    TRACKEVAL_INPUTS_GT / "MOT17-train" / "MOT17-02-DPM" / "det" / "det.txt",
    TRACKEVAL_INPUTS_GT / "MOT17-train" / "MOT17-04-DPM" / "det" / "det.txt",
    TRACKEVAL_INPUTS_TRACKERS / "MOT17-train" / "yolov8n_2025-08-31_21-10-38" / "data" / "MOT17-02-DPM.txt",
    TRACKEVAL_INPUTS_TRACKERS / "MOT17-train" / "yolov8n_2025-08-31_21-10-38" / "data" / "MOT17-04-DPM.txt",
    TRACKEVAL_INPUTS_TRACKERS / "MOT17-train" / "yolo11m_2025-08-31_21-10-38" / "data" / "MOT17-02-DPM.txt",
    TRACKEVAL_INPUTS_TRACKERS / "MOT17-train" / "yolo11m_2025-08-31_21-10-38" / "data" / "MOT17-04-DPM.txt"
]

def generate_TrackEval_detection_files(detection_file_glb_list):
    for detection_file in detection_file_glb_list:
        detection_file = Path(detection_file)

        if detection_file.resolve().is_relative_to(TRACKEVAL_INPUTS_TRACKERS):
            output_file = detection_file.parents[2] / f"{detection_file.parents[1].name}_for_eval" / detection_file.parent.name / detection_file.name
        elif detection_file.resolve().is_relative_to(TRACKEVAL_INPUTS_GT):
            output_file = TRACKEVAL_INPUTS_TRACKERS / detection_file.parents[2].name / "det_for_eval" / "data" / f"{detection_file.parents[1].name}.txt"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        out_lines = []
        current_frame = None
        local_id = 0
        with open(detection_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                frame = int(float(parts[0]))
                if frame != current_frame:
                    current_frame = frame
                    local_id = 0
                local_id += 1
                parts[1] = str(local_id)   # replace ID (-1) with increasing ID
                out_lines.append(",".join(parts))

        with open(output_file, "w") as f:
            f.write("\n".join(out_lines))

# generate_TrackEval_detection_files(detection_file_glb_list)

# For compatibility with numpy >= 1.24
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"):   np.int = int
if not hasattr(np, "bool"):  np.bool = bool

ROOT = Path(__file__).resolve().parent

sys.argv = [
    "run_mot_challenge.py",
    "--BENCHMARK", "MOT17",
    "--SPLIT_TO_EVAL", "train",
    "--TRACKERS_TO_EVAL", "yolo11m_osnet_x0_25_market1501_deepsort_2025-08-31_21-10-38", "yolov8n_2025-08-31_21-10-38_for_eval",
    "--METRICS", "HOTA", "CLEAR", "Identity", "VACE",
    "--USE_PARALLEL", "False",
    "--NUM_PARALLEL_CORES", "1",
    "--BREAK_ON_ERROR", "True",
    "--PRINT_RESULTS", "True",
    "--OUTPUT_SUB_FOLDER", "results",
    "--CLASSES_TO_EVAL", "pedestrian"
]

runpy.run_path(str(ROOT / "TrackEval" / "scripts" / "run_mot_challenge.py"), run_name="__main__")