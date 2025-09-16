import numpy as np
import torch
import cv2
import supervision as sv

def create_tracker(name, **kwargs):
    """
    Trackers supported:
      - deepsort       (deep_sort_realtime)  -> requires external embeddings in input of tracker_update (produced by extract_embeddings)
      - bytetrack      (supervision)         -> no ReID carried out, uses detections directly
      - botsort        (boxmot)              -> internal ReID carried out, requires reid weights in input
      - strongsort     (boxmot)              -> idem
      - deepocsort     (boxmot)              -> idem
      - hybridsort     (boxmot)              -> idem
      - boosttrack     (boxmot)              -> idem
    
    Returns:
      - Instantiated Tracker.
    """

    if name == 'deepsort':
        from deep_sort_realtime.deepsort_tracker import DeepSort
        return DeepSort(
            max_age=kwargs.get('max_age', 30),
            n_init=kwargs.get('n_init', 1),
            nn_budget=kwargs.get('nn_budget', 100),
            max_iou_distance=kwargs.get('max_iou_distance', 0.7)
        )

    elif name == 'bytetrack':
        return sv.ByteTrack()

    if name in {'botsort','strongsort','deepocsort','hybridsort','boosttrack'}:
        reidW = kwargs.get('reid_weights', None)
        common = dict(
            reid_weights=reidW,
            device=kwargs.get('device', 'cuda:0'),
            half=kwargs.get('half', True)
        )

        if name == 'botsort':
            from boxmot import BotSort
            return BotSort(
                track_high_thresh=kwargs.get('track_high_thresh', 0.5),
                new_track_thresh=kwargs.get('new_track_thresh', 0.6),
                track_buffer=kwargs.get('track_buffer', 30),
                match_thresh=kwargs.get('match_thresh', 0.8),
                **common
            )
        if name == 'strongsort':
            from boxmot import StrongSort
            return StrongSort(
                min_conf=kwargs.get('min_conf', 0.1),
                max_cos_dist=kwargs.get('max_cos_dist', 0.2),
                max_iou_dist=kwargs.get('max_iou_dist', 0.7),
                max_age=kwargs.get('max_age', 30),
                n_init=kwargs.get('n_init', 1),
                nn_budget=kwargs.get('nn_budget', 100),
                mc_lambda=kwargs.get('mc_lambda', 0.98),
                ema_alpha=kwargs.get('ema_alpha', 0.9),
                **common
            )
        if name == 'deepocsort':
            from boxmot import DeepOcSort
            return DeepOcSort(**common)
        if name == 'hybridsort':
            from boxmot import HybridSort
            return HybridSort(
                det_thresh=kwargs.get('det_thresh', 0.25),
                **common)
        if name == 'boosttrack':
            from boxmot import BoostTrack
            return BoostTrack(**common)

    raise ValueError(f"Unknown Tracker: {name}")

def xyxy_to_tlwh(x1, y1, x2, y2):

    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def tracker_update(tracker, name, boxes, confs, class_ids, img, embeds=None):
    """
    Returns:
    Unified output: list of (tid, x1, y1, x2, y2)
    """

    if name == 'deepsort':
        detections = []
        for i, (x1,y1,x2,y2) in enumerate(boxes):
            tlwh = xyxy_to_tlwh(x1,y1,x2,y2)
            feat = None if (embeds is None or len(embeds)==0) else embeds[i].astype(np.float32)
            detections.append((tlwh, float(confs[i]), feat))
        tracks = tracker.update_tracks(detections, frame=img)

        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1,y1,x2,y2 = t.to_tlbr()
            out.append((int(t.track_id), float(x1), float(y1), float(x2), float(y2)))
        return out

    if name == 'bytetrack':
        if len(boxes) == 0:
            dets = sv.Detections.empty()
        else:
            dets = sv.Detections(
                xyxy=np.asarray(boxes, dtype=float),
                confidence=np.asarray(confs, dtype=float)
            )
        tracks = tracker.update_with_detections(dets) # Update on the current frame

        out = []
        if tracks is not None and len(tracks) > 0:
            for (x1,y1,x2,y2), tid in zip(tracks.xyxy, tracks.tracker_id):
                if tid is not None:
                    out.append((int(tid), float(x1), float(y1), float(x2), float(y2)))
        return out # Tracking data for the current frame

    if name in {'botsort','strongsort','deepocsort','hybridsort','boosttrack'}:
        if len(boxes) == 0:
            dets = np.zeros((0,6), dtype=float)
        else:
            dets = np.hstack([
                np.array(boxes, dtype=float),
                np.array(confs, dtype=float).reshape(-1,1),
                np.array(class_ids, dtype=float).reshape(-1, 1)
            ])

        tracks = tracker.update(dets, img) # Update on the current frame
        if tracks is None:
            return []
        tracks = np.asarray(tracks)
        if tracks.size == 0:
            return []
        if tracks.ndim == 1:
            if tracks.shape[0] < 5:
                return []
            tracks = tracks.reshape(1, -1)

        out = []
        for x1,y1,x2,y2,tid in tracks[:, :5]:
            out.append((int(tid), float(x1), float(y1), float(x2), float(y2)))
        return out # Tracking data for the current frame

    raise ValueError(f"Unmanaged Tracker: {name}")

# Extraction of embeddings if DeepSort specified as tracker, via the specific ReID torchreid model informed, if it exists
def extract_embeddings(img, boxes, reid_model, device):
    """
    Extracts embeddings from the given image and bounding boxes using the specified ReID model.
    """
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(int(img.shape[1]), x2)
        y2 = min(int(img.shape[0]), y2)
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 256)) # Resize (W, H) (size expected by torchreid models)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) # OpenCV → RGB (torchreid models expect RGB)
        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0 # Convert to tensor and normalize
        crop = (crop - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
               torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) # Normalization ImageNet
        crops.append(crop)

    if not crops:
        return []
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        embeddings = reid_model(batch)
    return embeddings.cpu().numpy()