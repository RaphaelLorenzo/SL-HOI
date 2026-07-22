from dataclasses import dataclass
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-hoister"))
try:
    import supervision as sv
except ImportError:
    class _FallbackDetections:
        def __init__(self, xyxy, confidence=None, class_id=None, tracker_id=None):
            self.xyxy = xyxy
            self.confidence = confidence
            self.class_id = class_id
            self.tracker_id = tracker_id

        def __len__(self):
            return len(self.xyxy)

    class _FallbackByteTrack:
        def __init__(self, *args, **kwargs):
            pass

        def update_with_detections(self, detections):
            return _FallbackDetections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id,
                tracker_id=None,
            )

    class _FallbackSupervision:
        Detections = _FallbackDetections
        ByteTrack = _FallbackByteTrack

    sv = _FallbackSupervision()


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(
        a[:, 3] - a[:, 1], 0.0, None
    )
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(
        b[:, 3] - b[:, 1], 0.0, None
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, a_min=1e-6, a_max=None)


@dataclass
class PersonColorTracker:
    frame_rate: int
    max_people: int = 2
    track_activation_threshold: float = 0.05
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8

    def __post_init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=max(1, int(self.frame_rate)),
        )
        self.track_slots = {}
        self.missing_by_track = {}

    def _release_missing_slots(self, seen_track_ids):
        for track_id in list(self.track_slots.keys()):
            if track_id in seen_track_ids:
                self.missing_by_track[track_id] = 0
                continue
            self.missing_by_track[track_id] = (
                int(self.missing_by_track.get(track_id, 0)) + 1
            )
            if int(self.missing_by_track[track_id]) > int(self.lost_track_buffer):
                del self.track_slots[track_id]
                del self.missing_by_track[track_id]

    def _assign_slot_for_track(self, track_id):
        track_id = int(track_id)
        if track_id in self.track_slots:
            self.missing_by_track[track_id] = 0
            return int(self.track_slots[track_id])

        used_slots = set(int(slot) for slot in self.track_slots.values())
        for slot in range(max(0, int(self.max_people))):
            if slot not in used_slots:
                self.track_slots[track_id] = slot
                self.missing_by_track[track_id] = 0
                return slot
        return None

    def assign_pair_color_indices(self, pairs):
        if not pairs:
            self._release_missing_slots(set())
            return {}

        person_items = {}
        for pair in pairs:
            person_idx = int(pair["person_idx"])
            if person_idx not in person_items:
                person_items[person_idx] = {
                    "box": pair["person_box"],
                    "score": pair.get("person_score", 1.0),
                }

        person_indices = list(person_items.keys())
        boxes = np.asarray(
            [person_items[idx]["box"] for idx in person_indices], dtype=np.float32
        )
        scores = np.asarray(
            [
                1.0 if person_items[idx]["score"] is None else person_items[idx]["score"]
                for idx in person_indices
            ],
            dtype=np.float32,
        )

        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=np.zeros(len(boxes), dtype=np.int32),
        )
        tracked = self.tracker.update_with_detections(detections)

        person_to_color_idx = {}
        seen_track_ids = set()
        if tracked.tracker_id is not None and len(tracked) > 0:
            ious = _box_iou_matrix(boxes, tracked.xyxy.astype(np.float32))
            used_tracked = set()
            for person_pos, person_idx in enumerate(person_indices):
                order = np.argsort(-ious[person_pos])
                for tracked_pos in order:
                    if tracked_pos in used_tracked:
                        continue
                    if ious[person_pos, tracked_pos] <= 0.0:
                        break
                    used_tracked.add(int(tracked_pos))
                    tracker_id = int(tracked.tracker_id[tracked_pos])
                    slot = self._assign_slot_for_track(tracker_id)
                    seen_track_ids.add(tracker_id)
                    if slot is not None:
                        person_to_color_idx[person_idx] = int(slot)
                    break

        self._release_missing_slots(seen_track_ids)
        used_slots = set(person_to_color_idx.values())
        for person_idx in person_indices:
            if person_idx in person_to_color_idx:
                continue
            for slot in range(max(0, int(self.max_people))):
                if slot not in used_slots:
                    person_to_color_idx[person_idx] = slot
                    used_slots.add(slot)
                    break

        pairs[:] = [
            pair
            for pair in pairs
            if int(pair["person_idx"]) in person_to_color_idx
        ]

        for pair in pairs:
            pair["color_idx"] = person_to_color_idx[int(pair["person_idx"])]
        return person_to_color_idx

    def assign_result_state_color_indices(self, result_state):
        if not result_state:
            return result_state

        pairs = result_state.get("pairs", [])
        person_to_color_idx = self.assign_pair_color_indices(pairs)
        for mask_item in result_state.get("summary_masks", []):
            person_idx = int(mask_item["person_idx"])
            if person_idx in person_to_color_idx:
                mask_item["color_idx"] = person_to_color_idx[person_idx]
        for verb_item in result_state.get("top_verb_predictions", []):
            person_idx = int(verb_item["person_idx"])
            if person_idx in person_to_color_idx:
                verb_item["color_idx"] = person_to_color_idx[person_idx]
        return result_state


@dataclass
class ObjectClassStabilizer:
    switch_frames: int = 3
    iou_threshold: float = 0.35
    max_missing_frames: int = 15

    def __post_init__(self):
        self.tracks = {}
        self.next_track_id = 0

    @staticmethod
    def _pair_person_key(pair):
        return int(pair.get("color_idx", pair.get("person_idx", -1)))

    def _new_track(self, pair):
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = {
            "box": tuple(map(float, pair["object_box"])),
            "person_key": self._pair_person_key(pair),
            "stable_obj_id": int(pair["object_id"]),
            "pending_obj_id": None,
            "pending_count": 0,
            "missing": 0,
        }
        return track_id

    def _match_pairs_to_tracks(self, pairs):
        if not pairs or not self.tracks:
            return {}

        pair_boxes = np.asarray([pair["object_box"] for pair in pairs], dtype=np.float32)
        track_ids = list(self.tracks.keys())
        track_boxes = np.asarray(
            [self.tracks[track_id]["box"] for track_id in track_ids],
            dtype=np.float32,
        )
        ious = _box_iou_matrix(pair_boxes, track_boxes)

        candidates = []
        for pair_idx, pair in enumerate(pairs):
            person_key = self._pair_person_key(pair)
            for track_pos, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                if int(track["person_key"]) != person_key:
                    continue
                iou = float(ious[pair_idx, track_pos])
                if iou >= float(self.iou_threshold):
                    candidates.append((iou, pair_idx, track_id))

        candidates.sort(reverse=True)
        matched_pairs = set()
        matched_tracks = set()
        pair_to_track = {}
        for _iou, pair_idx, track_id in candidates:
            if pair_idx in matched_pairs or track_id in matched_tracks:
                continue
            matched_pairs.add(pair_idx)
            matched_tracks.add(track_id)
            pair_to_track[pair_idx] = track_id
        return pair_to_track

    def update_result_state(self, result_state):
        if not result_state:
            return result_state

        pairs = result_state.get("pairs", [])
        pair_to_track = self._match_pairs_to_tracks(pairs)
        matched_track_ids = set(pair_to_track.values())

        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_track_ids:
                track["missing"] += 1
                if int(track["missing"]) > int(self.max_missing_frames):
                    del self.tracks[track_id]

        for pair_idx, pair in enumerate(pairs):
            track_id = pair_to_track.get(pair_idx)
            if track_id is None:
                track_id = self._new_track(pair)
            track = self.tracks[track_id]

            predicted_obj_id = int(pair["object_id"])
            stable_obj_id = int(track["stable_obj_id"])
            if predicted_obj_id == stable_obj_id:
                track["pending_obj_id"] = None
                track["pending_count"] = 0
            else:
                if track["pending_obj_id"] == predicted_obj_id:
                    track["pending_count"] += 1
                else:
                    track["pending_obj_id"] = predicted_obj_id
                    track["pending_count"] = 1

                if int(track["pending_count"]) >= int(self.switch_frames):
                    track["stable_obj_id"] = predicted_obj_id
                    track["pending_obj_id"] = None
                    track["pending_count"] = 0

            track["box"] = tuple(map(float, pair["object_box"]))
            track["person_key"] = self._pair_person_key(pair)
            track["missing"] = 0

            pair["object_track_id"] = int(track_id)
            pair["raw_object_id"] = int(predicted_obj_id)
            pair["stable_object_id"] = int(track["stable_obj_id"])

        return result_state


@dataclass
class InteractionStabilizer:
    alpha: float = 0.4
    exit_ratio: float = 0.65
    max_missing_frames: int = 8

    def __post_init__(self):
        self.tracks = {}

    @staticmethod
    def _pair_person_key(pair):
        return int(pair.get("color_idx", pair.get("person_idx", -1)))

    @staticmethod
    def _pair_target_key(pair):
        if pair.get("object_track_id") is not None:
            return ("track", int(pair["object_track_id"]))
        if pair.get("visual_group_id") is not None:
            return ("visual", int(pair["visual_group_id"]))
        return ("query", int(pair.get("query_idx", -1)), int(pair.get("object_id", -1)))

    @staticmethod
    def _interaction_key(item):
        if item.get("verb_only_mode"):
            return ("verb", int(item["verb_id"]))
        return ("hoi", int(item["verb_id"]), int(item.get("obj_id", -1)))

    def update_result_state(self, result_state, enter_threshold: float):
        if not result_state:
            return result_state

        exit_threshold = float(enter_threshold) * float(self.exit_ratio)
        seen_keys = set()
        pairs_out = []

        for pair in result_state.get("pairs", []):
            person_key = self._pair_person_key(pair)
            target_key = self._pair_target_key(pair)
            kept_items = []

            for item in pair.get("interactions", []):
                key = (person_key, target_key, self._interaction_key(item))
                raw_score = float(item.get("score", 0.0))
                track = self.tracks.get(key)
                if track is None:
                    smoothed_score = raw_score
                    visible = raw_score >= float(enter_threshold)
                    track = {
                        "score": smoothed_score,
                        "visible": visible,
                        "missing": 0,
                    }
                else:
                    smoothed_score = (
                        float(self.alpha) * raw_score
                        + (1.0 - float(self.alpha)) * float(track["score"])
                    )
                    visible = bool(track["visible"])
                    if visible:
                        visible = smoothed_score >= exit_threshold
                    else:
                        visible = smoothed_score >= float(enter_threshold)
                    track.update(
                        {
                            "score": smoothed_score,
                            "visible": visible,
                            "missing": 0,
                        }
                    )

                self.tracks[key] = track
                seen_keys.add(key)

                if not bool(track["visible"]):
                    continue

                item = dict(item)
                item["raw_score"] = raw_score
                item["score"] = float(track["score"])
                item["score_smoothed"] = True
                kept_items.append(item)

            if kept_items:
                pair = dict(pair)
                kept_items.sort(
                    key=lambda item: float(item.get("score", 0.0)),
                    reverse=True,
                )
                pair["interactions"] = kept_items
                pairs_out.append(pair)

        for key, track in list(self.tracks.items()):
            if key in seen_keys:
                continue
            track["missing"] = int(track.get("missing", 0)) + 1
            track["score"] = float(track.get("score", 0.0)) * (
                1.0 - float(self.alpha)
            )
            if float(track["score"]) < exit_threshold:
                track["visible"] = False
            if int(track["missing"]) > int(self.max_missing_frames):
                del self.tracks[key]

        pairs_by_visual_group = {}
        for pair in pairs_out:
            group_id = pair.get("visual_group_id")
            if group_id is None:
                continue
            pairs_by_visual_group.setdefault(int(group_id), []).append(pair)

        for group_pairs in pairs_by_visual_group.values():
            if any(pair.get("show_visual", True) for pair in group_pairs):
                continue
            rep = max(
                group_pairs,
                key=lambda pair: (
                    pair["interactions"][0]["score"] if pair.get("interactions") else 0.0
                ),
            )
            rep["show_visual"] = True

        result_state["pairs"] = pairs_out
        return result_state


@dataclass
class TargetColorStabilizer:
    max_missing_frames: int = 30

    def __post_init__(self):
        self.assignments = {}
        self.next_by_person = {}
        self.missing = {}

    @staticmethod
    def _pair_person_key(pair):
        return int(pair.get("color_idx", pair.get("person_idx", -1)))

    @staticmethod
    def _pair_target_key(pair):
        if pair.get("object_track_id") is not None:
            return ("track", int(pair["object_track_id"]))
        if pair.get("visual_group_id") is not None:
            return ("visual", int(pair["visual_group_id"]))
        return ("query", int(pair.get("query_idx", -1)), int(pair.get("object_id", -1)))

    def _assignment_key(self, pair):
        target_key = self._pair_target_key(pair)
        if target_key[0] == "track":
            return target_key
        return (self._pair_person_key(pair), target_key)

    def update_result_state(self, result_state):
        if not result_state:
            return result_state

        seen_keys = set()
        grouped_pairs = {}
        for pair in result_state.get("pairs", []):
            group_key = (
                self._pair_person_key(pair),
                int(pair.get("visual_group_id", pair.get("query_idx", -1))),
            )
            grouped_pairs.setdefault(group_key, []).append(pair)

        used_assignment_keys = set()
        used_shades_by_person = {}
        for group_pairs in grouped_pairs.values():
            representative = max(
                group_pairs,
                key=lambda pair: (
                    int(bool(pair.get("show_visual", True))),
                    pair["interactions"][0]["score"] if pair.get("interactions") else 0.0,
                ),
            )
            person_key = self._pair_person_key(representative)
            key = self._assignment_key(representative)
            if key in used_assignment_keys:
                key = (
                    person_key,
                    "visual",
                    int(representative.get("visual_group_id", representative.get("query_idx", -1))),
                )
            if key not in self.assignments:
                shade_idx = int(self.next_by_person.get(person_key, 0))
                used_shades = used_shades_by_person.setdefault(person_key, set())
                while shade_idx in used_shades:
                    shade_idx += 1
                self.assignments[key] = shade_idx
                self.next_by_person[person_key] = shade_idx + 1

            for pair in group_pairs:
                pair["target_color_idx"] = int(self.assignments[key])
            self.missing[key] = 0
            seen_keys.add(key)
            used_assignment_keys.add(key)
            used_shades_by_person.setdefault(person_key, set()).add(
                int(self.assignments[key])
            )

        for key in list(self.assignments.keys()):
            if key in seen_keys:
                continue
            self.missing[key] = int(self.missing.get(key, 0)) + 1
            if int(self.missing[key]) > int(self.max_missing_frames):
                del self.assignments[key]
                del self.missing[key]

        return result_state
