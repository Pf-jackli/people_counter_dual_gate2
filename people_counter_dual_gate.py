import argparse
from datetime import datetime
import math
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from ultralytics import YOLO

MAX_OCCLUSION_GAP_FRAMES = 180
RECENT_CROSSING_WINDOW_FRAMES = 5
PENDING_CROSSING_TIMEOUT_SECONDS = 10
STATIONARY_ALARM_SECONDS = 30
STATIONARY_TRACK_GAP_SECONDS = 3
STATIONARY_MOVEMENT_TOLERANCE = 30
LINE_DRAG_MARGIN = 30
LINE_OUTLINE_WIDTH = 8
LINE_WIDTH = 4
MIN_LINE_GAP = 30
MIN_GATE_LENGTH = 20


def bundled_resource_path(filename):
    bundle_dir = getattr(sys, "_MEIPASS", None)
    base_dir = Path(bundle_dir) if bundle_dir else Path(__file__).resolve().parent
    return base_dir / filename


def gate_assist_crop(frame, left_line, right_line):
    margin = max(128, round((right_line - left_line) * 1.1))
    left = max(0, left_line - margin)
    right = min(frame.shape[1], right_line + margin)
    return frame[:, left:right], left


def merge_gate_assist_detections(result, crop_result, left, crop_width, iou):
    full_boxes = result.boxes.data
    crop_boxes = crop_result.boxes.data.clone().to(full_boxes.device)
    if len(crop_boxes):
        valid = (
            ((crop_boxes[:, 0] > 4) | (left == 0))
            & (
                (crop_boxes[:, 2] < crop_width - 4)
                | (left + crop_width == result.orig_shape[1])
            )
        )
        crop_boxes = crop_boxes[valid]
        crop_boxes[:, (0, 2)] += left
    combined = torch.cat((full_boxes, crop_boxes), dim=0)
    if len(combined):
        combined = combined[nms(combined[:, :4], combined[:, 4], iou)]
    result.update(boxes=combined)


def add_gate_assist(model, assist_model, source, conf, iou, imgsz):
    def merge_before_tracking(predictor):
        if source.left_line is None or source.right_line is None:
            raise RuntimeError(f"{source.name}: 门线未设置，无法启用区域辅助识别")
        for result in predictor.results:
            crop, left = gate_assist_crop(
                result.orig_img, source.left_line, source.right_line
            )
            crop_results = assist_model.predict(
                crop, classes=[0], conf=conf, iou=iou, imgsz=imgsz,
                device=predictor.args.device, verbose=False,
            )
            if len(crop_results) != 1 or crop_results[0].boxes is None:
                raise RuntimeError(f"{source.name}: 区域辅助识别未返回检测结果")
            merge_gate_assist_detections(
                result, crop_results[0], left, crop.shape[1], iou
            )

    model.add_callback("on_predict_postprocess_end", merge_before_tracking)


@dataclass
class SourceState:
    name: str
    window_name: str
    capture: cv2.VideoCapture
    model: YOLO | None = None
    left_line: int | None = None
    right_line: int | None = None
    fps: float = 50.0
    frame_width: int | None = None
    frame_height: int | None = None
    dragging_line: str | None = None
    setup_message: str = ""
    window_created: bool = False
    video_files: list = field(default_factory=list)
    video_index: int = 0
    last_center_x: dict = field(default_factory=dict)
    last_track_boxes: dict = field(default_factory=dict)
    track_history: dict = field(default_factory=dict)
    pending_directions: dict = field(default_factory=dict)
    counted_track_directions: set = field(default_factory=set)
    recent_crossings: list = field(default_factory=list)
    recent_gate_starts: list = field(default_factory=list)
    nested_track_pairs: dict = field(default_factory=dict)
    separated_track_pairs: dict = field(default_factory=dict)
    visible_track_boxes: dict = field(default_factory=dict)
    visible_tracks_frame: int = -1
    stationary_tracks: dict = field(default_factory=dict)
    stationary_alerted_ids: set = field(default_factory=set)
    left_gate_y: tuple[int, int] | None = None
    right_gate_y: tuple[int, int] | None = None
    gate_setup_target: str | None = None
    gate_click_points: dict = field(
        default_factory=lambda: {"left": [], "right": []}
    )
    frame_index: int = 0
    debug_frame_context: str = ""
    entry_direction: str = "left-to-right"
    count_in: int = 0
    count_out: int = 0
    active: bool = True


def seconds_to_frames(source, seconds):
    return max(1, round(source.fps * seconds))


def make_debug_frame_context(source):
    wall_time = datetime.now().astimezone().isoformat(timespec="milliseconds")
    if not source.video_files:
        return f"处理时间={wall_time} 帧={source.frame_index} 视频位置=不适用（实时源）"

    video_name = source.video_files[source.video_index].name
    frame_position = source.capture.get(cv2.CAP_PROP_POS_FRAMES)
    if math.isfinite(frame_position) and frame_position >= 1:
        video_frame = str(int(frame_position) - 1)
    else:
        video_frame = "不可用"

    position_ms = source.capture.get(cv2.CAP_PROP_POS_MSEC)
    if math.isfinite(position_ms) and position_ms >= 0:
        milliseconds = round(position_ms)
        hours, milliseconds = divmod(milliseconds, 3_600_000)
        minutes, milliseconds = divmod(milliseconds, 60_000)
        seconds, milliseconds = divmod(milliseconds, 1_000)
        video_position = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    else:
        video_position = "不可用"

    return (
        f"处理时间={wall_time} 视频文件={video_name} "
        f"文件帧号（从0开始）={video_frame} "
        f"解码器报告视频位置={video_position} 帧={source.frame_index}"
    )


def log_debug_track_boxes(source, ids, coordinates):
    tracks = []
    for track_id, box in zip(ids, coordinates):
        x1, y1, x2, y2 = map(int, box)
        tracks.append(f"ID:{track_id}框=({x1},{y1},{x2},{y2})")
    if tracks:
        print(
            f"\n[逐帧跟踪] {source.name} {source.debug_frame_context} "
            + " ".join(tracks)
        )


def update_nested_track_pairs(source, ids, coordinates):
    tracks = [(track_id, tuple(map(int, box))) for track_id, box in zip(ids, coordinates)]
    source.visible_track_boxes = dict(tracks)
    source.visible_tracks_frame = source.frame_index
    previous_pairs = source.nested_track_pairs
    visible_ids = {track_id for track_id, _ in tracks}
    pending_timeout = seconds_to_frames(source, PENDING_CROSSING_TIMEOUT_SECONDS)
    source.separated_track_pairs = {
        pair: last_frame
        for pair, last_frame in source.separated_track_pairs.items()
        if source.frame_index - last_frame <= pending_timeout
        and any(track_id in source.pending_directions for track_id in pair)
    }
    current_pairs = {
        pair: (count, last_frame)
        for pair, (count, last_frame) in previous_pairs.items()
        if count >= 3
        and source.frame_index - last_frame
        <= (
            pending_timeout
            if any(track_id in source.pending_directions for track_id in pair)
            else 10
        )
        and not all(track_id in visible_ids for track_id in pair)
    }
    for index, (first_id, first_box) in enumerate(tracks):
        for second_id, second_box in tracks[index + 1:]:
            if first_id in source.pending_directions or second_id in source.pending_directions:
                padding = max(
                    20,
                    min(first_box[2] - first_box[0], second_box[2] - second_box[0]) * 0.15,
                )
                if (
                    first_box[2] + padding < second_box[0]
                    or second_box[2] + padding < first_box[0]
                ):
                    source.separated_track_pairs[tuple(sorted((first_id, second_id)))] = (
                        source.frame_index
                    )
            first_width = first_box[2] - first_box[0]
            first_height = first_box[3] - first_box[1]
            second_width = second_box[2] - second_box[0]
            second_height = second_box[3] - second_box[1]
            first_area = first_width * first_height
            second_area = second_width * second_height
            if min(first_width, first_height, second_width, second_height) <= 0:
                continue
            if first_area <= second_area:
                small, large = first_box, second_box
                small_area, large_area = first_area, second_area
            else:
                small, large = second_box, first_box
                small_area, large_area = second_area, first_area
            intersection = max(0, min(small[2], large[2]) - max(small[0], large[0])) * max(
                0, min(small[3], large[3]) - max(small[1], large[1])
            )
            if (
                intersection / small_area < 0.9
                or small_area / large_area > 0.6
                or abs(small[1] - large[1]) > (large[3] - large[1]) * 0.2
                or (small[2] - small[0]) < (large[2] - large[0]) * 0.3
            ):
                continue
            pair = tuple(sorted((first_id, second_id)))
            count, previous_frame = previous_pairs.get(pair, (0, -1))
            current_pairs[pair] = (
                count + 1
                if previous_frame == source.frame_index - 1
                or (count >= 3 and source.frame_index - previous_frame <= 10)
                else 1,
                source.frame_index,
            )
    source.nested_track_pairs = current_pairs


def video_frame_rate(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    return fps if fps > 0 else 50.0


def is_inside_channel(source, center_x, foot_y, left_line_x, right_line_x):
    if not left_line_x < center_x < right_line_x:
        return False
    if source.left_gate_y is None or source.right_gate_y is None:
        return True

    progress = (center_x - left_line_x) / (right_line_x - left_line_x)
    upper_y = source.left_gate_y[0] + progress * (
        source.right_gate_y[0] - source.left_gate_y[0]
    )
    lower_y = source.left_gate_y[1] + progress * (
        source.right_gate_y[1] - source.left_gate_y[1]
    )
    return upper_y <= foot_y <= lower_y


def update_stationary_alarm(source, track_id, box, left_line_x, right_line_x):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    foot_y = y2
    stale_after = seconds_to_frames(source, STATIONARY_TRACK_GAP_SECONDS)
    expired_ids = [
        tracked_id
        for tracked_id, state in source.stationary_tracks.items()
        if source.frame_index - state[3] > stale_after
    ]
    for expired_id in expired_ids:
        source.stationary_tracks.pop(expired_id, None)
        source.stationary_alerted_ids.discard(expired_id)

    if not is_inside_channel(source, center_x, foot_y, left_line_x, right_line_x):
        source.stationary_tracks.pop(track_id, None)
        source.stationary_alerted_ids.discard(track_id)
        return False

    movement_tolerance = max(
        STATIONARY_MOVEMENT_TOLERANCE, (y2 - y1) * 0.15
    )
    state = source.stationary_tracks.get(track_id)
    if state is None or (
        source.frame_index - state[3] > stale_after
        or np.hypot(center_x - state[1], foot_y - state[2]) > movement_tolerance
    ):
        source.stationary_tracks[track_id] = (
            source.frame_index,
            center_x,
            foot_y,
            source.frame_index,
        )
        source.stationary_alerted_ids.discard(track_id)
        return False

    source.stationary_tracks[track_id] = (
        state[0],
        state[1],
        state[2],
        source.frame_index,
    )
    if (
        track_id not in source.stationary_alerted_ids
        and source.frame_index - state[0] >= seconds_to_frames(
            source, STATIONARY_ALARM_SECONDS
        )
    ):
        source.stationary_alerted_ids.add(track_id)
        print(
            f"\n[异常滞留报警] {source.name} ID:{track_id} "
            f"在两道计数线之间约 {STATIONARY_ALARM_SECONDS} 秒位置变化较小，"
            "请人工核查（仅为滞留提示，不代表已确认摔倒或昏厥）"
        )
        return True
    return False


def crossing_y_at_line(previous_x, previous_y, current_x, current_y, line_x):
    delta_x = current_x - previous_x
    if delta_x == 0:
        return None
    ratio = (line_x - previous_x) / delta_x
    if not 0 <= ratio <= 1:
        return None
    return previous_y + ratio * (current_y - previous_y)


def crosses_finite_gate(previous_x, current_x, previous_y, current_y, line_x, gate_y):
    if gate_y is None:
        return True
    if previous_y is None or current_y is None:
        return False
    crossing_y = crossing_y_at_line(
        previous_x, previous_y, current_x, current_y, line_x
    )
    return crossing_y is not None and gate_y[0] <= crossing_y <= gate_y[1]


def gate_transition(
    previous_center_x,
    current_center_x,
    left_line_x,
    right_line_x,
    previous_foot_y=None,
    current_foot_y=None,
    left_gate_y=None,
    right_gate_y=None,
):
    left_crossing_valid = crosses_finite_gate(
        previous_center_x,
        current_center_x,
        previous_foot_y,
        current_foot_y,
        left_line_x,
        left_gate_y,
    )
    right_crossing_valid = crosses_finite_gate(
        previous_center_x,
        current_center_x,
        previous_foot_y,
        current_foot_y,
        right_line_x,
        right_gate_y,
    )
    if previous_center_x < left_line_x:
        if current_center_x >= right_line_x:
            if left_crossing_valid and right_crossing_valid:
                return "through", "left-to-right"
            return None
        if current_center_x >= left_line_x:
            if left_crossing_valid:
                return "start", "left-to-right"
    elif previous_center_x > right_line_x:
        if current_center_x <= left_line_x:
            if left_crossing_valid and right_crossing_valid:
                return "through", "right-to-left"
            return None
        if current_center_x <= right_line_x:
            if right_crossing_valid:
                return "start", "right-to-left"
    elif left_line_x <= previous_center_x <= right_line_x:
        if current_center_x > right_line_x:
            if right_crossing_valid:
                return "complete", "left-to-right"
        if current_center_x < left_line_x:
            if left_crossing_valid:
                return "complete", "right-to-left"
    return None


def pending_tracks_in_box(
    box, frame_index, pending_directions, last_track_boxes, direction, current_track_id,
    separated_track_pairs,
):
    x1, y1, x2, y2 = box
    padding_x = max(20, (x2 - x1) * 0.15)
    padding_y = max(20, (y2 - y1) * 0.1)
    overlapping_ids = []
    for track_id, pending_direction in pending_directions.items():
        if pending_direction != direction:
            continue
        if track_id == current_track_id:
            overlapping_ids.append(track_id)
            continue
        if tuple(sorted((track_id, current_track_id))) in separated_track_pairs:
            continue
        previous_box = last_track_boxes.get(track_id)
        if previous_box is None:
            continue
        px1, py1, px2, py2, last_seen = previous_box
        if frame_index - last_seen > MAX_OCCLUSION_GAP_FRAMES:
            continue
        center_x = (px1 + px2) / 2
        center_y = (py1 + py2) / 2
        if (
            x1 - padding_x <= center_x <= x2 + padding_x
            and y1 - padding_y <= center_y <= y2 + padding_y
        ):
            overlapping_ids.append(track_id)
    return overlapping_ids


def motion_conflicts_with_direction(previous_center_x, current_center_x, direction):
    if previous_center_x is None:
        return False
    displacement = current_center_x - previous_center_x
    if direction == "left-to-right":
        return displacement < -2
    return displacement > 2


def pending_tracks_across_gate(
    source,
    box,
    current_center_x,
    left_line_x,
    right_line_x,
    direction,
    current_track_id,
    debug_counting=False,
):
    _, y1, _, y2 = box
    current_center_y = (y1 + y2) / 2
    current_height = y2 - y1
    current_foot_y = y2
    candidates = []

    for track_id, pending_direction in source.pending_directions.items():
        if track_id == current_track_id or pending_direction != direction:
            continue
        previous_box = source.last_track_boxes.get(track_id)
        previous_center_x = source.last_center_x.get(track_id)
        if previous_box is None or previous_center_x is None:
            continue

        _, py1, _, py2, last_seen = previous_box
        frame_gap = source.frame_index - last_seen
        if frame_gap <= 0 or frame_gap > MAX_OCCLUSION_GAP_FRAMES:
            continue
        if tuple(sorted((track_id, current_track_id))) in source.separated_track_pairs:
            if debug_counting:
                print(
                    f"\n[计数诊断] {source.name} ID:{current_track_id} "
                    f"跨ID候选ID:{track_id}关联拒绝：两条轨迹曾同帧分离"
                )
            continue
        visible_box = (
            source.visible_track_boxes.get(track_id)
            if source.visible_tracks_frame == source.frame_index
            else None
        )
        if visible_box is not None:
            padding = max(20, (box[2] - box[0]) * 0.15)
            if visible_box[2] < box[0] - padding or visible_box[0] > box[2] + padding:
                if debug_counting:
                    print(
                        f"\n[计数诊断] {source.name} ID:{current_track_id} "
                        f"跨ID候选ID:{track_id}关联拒绝：两条轨迹同帧可见且位置分离"
                    )
                continue
        previous_height = py2 - py1
        max_displacement = max(120, previous_height, current_height) + frame_gap * max(
            5, max(previous_height, current_height) * 0.08
        )
        horizontal_distance = abs(current_center_x - previous_center_x)
        if horizontal_distance > max_displacement:
            if debug_counting:
                print(
                    f"\n[计数诊断] {source.name} ID:{current_track_id} "
                    f"跨ID候选ID:{track_id}关联拒绝：相隔{frame_gap}帧"
                    f"水平位移{horizontal_distance:.1f}px超过"
                    f"允许范围{max_displacement:.1f}px"
                )
            continue
        exit_line_x = (
            right_line_x if direction == "left-to-right" else left_line_x
        )
        max_exit_overshoot = max(120, current_height * 1.5)
        if abs(current_center_x - exit_line_x) > max_exit_overshoot:
            if debug_counting:
                print(
                    f"\n[计数诊断] {source.name} ID:{current_track_id} "
                    f"跨ID候选ID:{track_id}关联拒绝：新轨迹位置"
                    f"x={current_center_x:.1f}超出目标门线"
                    f"x={exit_line_x}允许范围"
                    f"{max_exit_overshoot:.1f}px"
                )
            continue
        transition = gate_transition(
            previous_center_x,
            current_center_x,
            left_line_x,
            right_line_x,
            previous_box[3],
            current_foot_y,
            source.left_gate_y,
            source.right_gate_y,
        )
        if (
            transition is None
            or transition[0] not in ("complete", "through")
            or transition[1] != direction
        ):
            continue

        previous_center_y = (py1 + py2) / 2
        vertical_distance = abs(previous_center_y - current_center_y)
        vertical_tolerance = max(40, ((py2 - py1) + current_height) * 0.75)
        if vertical_distance <= vertical_tolerance:
            candidates.append((vertical_distance, track_id))

    if not candidates:
        return []
    candidates.sort()
    if len(candidates) > 1:
        separation_margin = max(20, current_height * 0.25)
        if candidates[1][0] - candidates[0][0] < separation_margin:
            return []
    return [candidates[0][1]]


def pending_tracks_between_gates(
    source, direction, current_center_x, left_line_x, right_line_x, current_track_id
):
    pending_timeout = seconds_to_frames(source, PENDING_CROSSING_TIMEOUT_SECONDS)
    candidates = []
    for track_id, pending_direction in source.pending_directions.items():
        if pending_direction != direction:
            continue
        if tuple(sorted((track_id, current_track_id))) in source.separated_track_pairs:
            continue
        previous_center_x = source.last_center_x.get(track_id)
        previous_box = source.last_track_boxes.get(track_id)
        if previous_center_x is None or previous_box is None:
            continue
        if source.frame_index - previous_box[4] > pending_timeout:
            continue
        if not left_line_x <= previous_center_x <= right_line_x:
            continue
        if direction == "left-to-right" and previous_center_x < current_center_x:
            candidates.append(track_id)
        elif direction == "right-to-left" and previous_center_x > current_center_x:
            candidates.append(track_id)
    return candidates


def estimate_path_y_at_x(history, target_x):
    if len(history) < 2:
        return None
    xs = [sample[0] for sample in history]
    ys = [sample[1] for sample in history]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    if variance_x < 25:
        return None
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / variance_x
    return mean_y + slope * (target_x - mean_x)


def recent_gate_start_matches_track(source, direction, track_id, exit_y, box):
    if exit_y is None:
        return None
    current_height = box[3] - box[1]
    pending_timeout = seconds_to_frames(source, PENDING_CROSSING_TIMEOUT_SECONDS)
    matches = []
    for (
        start_frame,
        start_direction,
        start_track_id,
        _start_y,
        start_height,
        expected_exit_y,
    ) in source.recent_gate_starts:
        if (
            start_direction != direction
            or start_track_id == track_id
            or tuple(sorted((start_track_id, track_id))) in source.separated_track_pairs
            or expected_exit_y is None
            or source.frame_index - start_frame > pending_timeout
        ):
            continue
        tolerance = max(80, (start_height + current_height) * 0.5)
        distance = abs(exit_y - expected_exit_y)
        if distance <= tolerance:
            matches.append((distance, start_frame, start_track_id))
    if not matches:
        return None
    matches.sort()
    if len(matches) > 1 and matches[1][0] - matches[0][0] < max(
        20, current_height * 0.2
    ):
        return None
    return matches[0][2]


def is_recent_duplicate_crossing(
    source,
    direction,
    box,
    window_frames=RECENT_CROSSING_WINDOW_FRAMES,
    start_frame=None,
    start_y=None,
    exit_y=None,
    person_height=None,
):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    source.recent_crossings = [
        crossing
        for crossing in source.recent_crossings
        if source.frame_index - crossing[0] <= MAX_OCCLUSION_GAP_FRAMES
    ]

    for crossing in source.recent_crossings:
        (
            crossing_frame,
            previous_direction,
            previous_box,
            previous_start_frame,
            previous_start_y,
            previous_exit_y,
            previous_height,
        ) = crossing
        if previous_direction != direction:
            continue
        px1, py1, px2, py2 = previous_box
        previous_center_x = (px1 + px2) / 2
        previous_center_y = (py1 + py2) / 2
        horizontal_tolerance = max(60, ((px2 - px1) + width) * 0.75)
        vertical_tolerance = max(60, ((py2 - py1) + height) * 0.75)
        same_recent_box = (
            source.frame_index - crossing_frame <= window_frames
            and abs(previous_center_x - center_x) <= horizontal_tolerance
            and abs(previous_center_y - center_y) <= vertical_tolerance
        )
        path_tolerance = max(30, ((previous_height or height) + (person_height or height)) * 0.4)
        same_recent_path = (
            start_frame is not None
            and previous_start_frame is not None
            and abs(start_frame - previous_start_frame)
            <= RECENT_CROSSING_WINDOW_FRAMES
            and start_y is not None
            and previous_start_y is not None
            and abs(start_y - previous_start_y) <= path_tolerance
            and exit_y is not None
            and previous_exit_y is not None
            and abs(exit_y - previous_exit_y) <= path_tolerance
            and abs(previous_center_y - center_y) <= vertical_tolerance
        )
        if same_recent_box or same_recent_path:
            return True

    source.recent_crossings.append(
        (
            source.frame_index,
            direction,
            (x1, y1, x2, y2),
            start_frame,
            start_y,
            exit_y,
            person_height,
        )
    )
    return False


def update_track_crossing(
    source, track_id, box, left_line_x, right_line_x, debug_counting=False
):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    update_stationary_alarm(
        source, track_id, (x1, y1, x2, y2), left_line_x, right_line_x
    )
    previous_box = source.last_track_boxes.get(track_id)
    previous_center_x = source.last_center_x.get(track_id)
    source.track_history.setdefault(track_id, []).append((center_x, y2))
    source.track_history[track_id] = source.track_history[track_id][-12:]
    pending_timeout = seconds_to_frames(source, PENDING_CROSSING_TIMEOUT_SECONDS)
    source.recent_gate_starts = [
        start
        for start in source.recent_gate_starts
        if source.frame_index - start[0] <= pending_timeout
    ]
    if debug_counting and previous_center_x is None:
        print(
            f"\n[计数诊断] {source.name} 首次检测 ID:{track_id} "
            f"x={center_x:.1f}，绿线x={left_line_x}，红线x={right_line_x} "
            f"框=({x1},{y1},{x2},{y2}) {source.debug_frame_context}"
        )
    expired_ids = [
        pending_id
        for pending_id in source.pending_directions
        if pending_id != track_id
        and (
            pending_id not in source.last_track_boxes
            or source.frame_index - source.last_track_boxes[pending_id][4]
            > pending_timeout
        )
    ]
    for pending_id in expired_ids:
        source.pending_directions.pop(pending_id, None)
        source.last_center_x.pop(pending_id, None)
        source.last_track_boxes.pop(pending_id, None)

    transition = None
    if previous_center_x is not None:
        transition = gate_transition(
            previous_center_x,
            center_x,
            left_line_x,
            right_line_x,
            previous_box[3] if previous_box is not None else None,
            y2,
            source.left_gate_y,
            source.right_gate_y,
        )
    pending_direction = source.pending_directions.get(track_id)
    crossed_exit_line_outside_gate = False
    if (
        previous_center_x is not None
        and previous_box is not None
        and pending_direction is not None
    ):
        crossed_exit_line_outside_gate = (
            pending_direction == "left-to-right"
            and previous_center_x <= right_line_x < center_x
            and not crosses_finite_gate(
                previous_center_x,
                center_x,
                previous_box[3],
                y2,
                right_line_x,
                source.right_gate_y,
            )
        ) or (
            pending_direction == "right-to-left"
            and previous_center_x >= left_line_x > center_x
            and not crosses_finite_gate(
                previous_center_x,
                center_x,
                previous_box[3],
                y2,
                left_line_x,
                source.left_gate_y,
            )
        )
        if crossed_exit_line_outside_gate:
            source.pending_directions.pop(track_id, None)
            if debug_counting:
                exit_line_x = (
                    right_line_x
                    if pending_direction == "left-to-right"
                    else left_line_x
                )
                exit_gate_y = (
                    source.right_gate_y
                    if pending_direction == "left-to-right"
                    else source.left_gate_y
                )
                print(
                    f"\n[计数诊断] {source.name} ID:{track_id} "
                    f"首门待计状态取消：脚点未穿过有限门段"
                    f"（门线x={exit_line_x}，门段y={exit_gate_y}）"
                )
    if (
        transition is not None
        and transition[0] == "complete"
        and source.pending_directions.get(track_id) not in (None, transition[1])
    ):
        source.pending_directions.pop(track_id, None)
    if transition is not None and transition[0] == "start":
        source.pending_directions[track_id] = transition[1]
        start_line_x = (
            left_line_x
            if transition[1] == "left-to-right"
            else right_line_x
        )
        start_y = None
        if previous_box is not None:
            start_y = crossing_y_at_line(
                previous_center_x,
                previous_box[3],
                center_x,
                y2,
                start_line_x,
            )
        if start_y is not None:
            target_line_x = (
                right_line_x
                if transition[1] == "left-to-right"
                else left_line_x
            )
            expected_exit_y = estimate_path_y_at_x(
                source.track_history[track_id], target_line_x
            )
            source.recent_gate_starts.append(
                (
                    source.frame_index,
                    transition[1],
                    track_id,
                    start_y,
                    y2 - y1,
                    expected_exit_y,
                )
            )
    if debug_counting and transition is not None:
        print(
            f"\n[计数诊断] {source.name} ID:{track_id} "
            f"x={previous_center_x:.1f}->{center_x:.1f} "
            f"状态={transition[0]} 方向={transition[1]} "
            f"框=({x1},{y1},{x2},{y2}) {source.debug_frame_context}"
        )

    completed = []
    for direction in ("left-to-right", "right-to-left"):
        inferred_missing_first_gate = False
        current_track_direction = source.pending_directions.get(track_id)
        motion_conflicts = motion_conflicts_with_direction(
            previous_center_x, center_x, direction
        )
        if current_track_direction is not None:
            candidate_ids = [track_id] if current_track_direction == direction else []
        elif (track_id, direction) in source.counted_track_directions:
            candidate_ids = []
        elif not motion_conflicts:
            candidate_ids = pending_tracks_in_box(
                (x1, y1, x2, y2),
                source.frame_index,
                source.pending_directions,
                source.last_track_boxes,
                direction,
                track_id,
                source.separated_track_pairs,
            )
        else:
            candidate_ids = []
        crossing_ids = []
        for candidate_id in candidate_ids:
            candidate_previous_x = source.last_center_x.get(candidate_id)
            if candidate_previous_x is None:
                continue
            candidate_transition = gate_transition(
                candidate_previous_x,
                center_x,
                left_line_x,
                right_line_x,
                source.last_track_boxes[candidate_id][3],
                y2,
                source.left_gate_y,
                source.right_gate_y,
            )
            if (
                candidate_transition is not None
                and candidate_transition[0] in ("complete", "through")
                and candidate_transition[1] == direction
            ):
                crossing_ids.append(candidate_id)

        if (
            transition is not None
            and transition[0] == "through"
            and transition[1] == direction
            and track_id not in crossing_ids
            and (track_id, direction) not in source.counted_track_directions
        ):
            crossing_ids.insert(0, track_id)

        if (
            not crossing_ids
            and current_track_direction is None
            and (track_id, direction) not in source.counted_track_directions
        ):
            if motion_conflicts:
                pending_ids = sorted(
                    pending_id
                    for pending_id, pending_direction in source.pending_directions.items()
                    if pending_direction == direction
                )
                if debug_counting and pending_ids:
                    print(
                        f"\n[计数诊断] {source.name} ID:{track_id} "
                        f"跨ID补配拒绝：新轨迹运动方向与待计方向相反"
                        f"（方向={direction}，待计ID={pending_ids}）"
                    )
            else:
                crossing_ids = pending_tracks_across_gate(
                    source,
                    (x1, y1, x2, y2),
                    center_x,
                    left_line_x,
                    right_line_x,
                    direction,
                    track_id,
                    debug_counting,
                )
        if (
            not crossing_ids
            and transition is not None
            and transition[0] == "complete"
            and transition[1] == direction
            and pending_direction is None
            and (track_id, direction) not in source.counted_track_directions
        ):
            active_first_gate_ids = pending_tracks_between_gates(
                source, direction, center_x, left_line_x, right_line_x, track_id
            )
            remembered_start_id = None
            if not active_first_gate_ids:
                remembered_start_id = recent_gate_start_matches_track(
                    source,
                    direction,
                    track_id,
                    crossing_y_at_line(
                        previous_center_x,
                        previous_box[3] if previous_box is not None else None,
                        center_x,
                        y2,
                        right_line_x
                        if direction == "left-to-right"
                        else left_line_x,
                    ),
                    (x1, y1, x2, y2),
                )
                if remembered_start_id is not None:
                    active_first_gate_ids = [remembered_start_id]
                elif debug_counting:
                    exit_y = crossing_y_at_line(
                        previous_center_x,
                        previous_box[3] if previous_box is not None else None,
                        center_x,
                        y2,
                        right_line_x
                        if direction == "left-to-right"
                        else left_line_x,
                    )
                    candidates = [
                        (start[2], round(start[5], 1))
                        for start in source.recent_gate_starts
                        if start[1] == direction
                        and start[2] != track_id
                        and source.frame_index - start[0]
                        <= pending_timeout
                        and start[5] is not None
                    ]
                    if candidates and exit_y is not None:
                        print(
                            f"\n[计数诊断] {source.name} ID:{track_id} "
                            f"首门轨迹匹配失败：第二门脚点y="
                            f"{exit_y:.1f}，候选首门轨迹及预测终点y={candidates}"
                        )
            if active_first_gate_ids:
                if debug_counting:
                    print(
                        f"\n[计数诊断] {source.name} ID:{track_id} "
                        f"与首门待计轨迹{active_first_gate_ids}关联，"
                        "按完整跨线计数"
                    )
                crossing_ids = active_first_gate_ids
                if remembered_start_id is not None:
                    source.recent_gate_starts = [
                        start
                        for start in source.recent_gate_starts
                        if not (
                            start[1] == direction
                            and start[2] == remembered_start_id
                        )
                    ]
            else:
                crossing_ids = [track_id]
                inferred_missing_first_gate = True
                if debug_counting:
                    print(
                        f"\n[计数诊断] {source.name} ID:{track_id} "
                        f"首门轨迹缺失，按本次完整跨越补计方向={direction}"
                    )
        if not crossing_ids:
            if (
                debug_counting
                and transition is not None
                and transition[0] == "complete"
                and transition[1] == direction
            ):
                pending_ids = sorted(
                    pending_id
                    for pending_id, pending_direction in source.pending_directions.items()
                    if pending_direction == direction
                )
                already_counted = (track_id, direction) in source.counted_track_directions
                reason = (
                    "该ID此方向已计数，跳过跨ID补配"
                    if already_counted
                    else "未找到方向一致的首门待计记录"
                )
                print(
                    f"\n[计数诊断] {source.name} ID:{track_id} "
                    f"完成第二道门但未计数：{reason}"
                    f"（本次轨迹原待计方向={pending_direction!r}，"
                    f"当前待计方向={source.pending_directions.get(track_id)!r}，"
                    f"第二道门外穿={crossed_exit_line_outside_gate}，"
                    f"可关联待计ID={pending_ids}，"
                    f"该ID此方向已计数={already_counted}）"
                )
            continue
        crossing_start = next(
            (
                start
                for start in reversed(source.recent_gate_starts)
                if start[1] == direction and start[2] in crossing_ids
            ),
            None,
        )
        exit_line_x = (
            right_line_x if direction == "left-to-right" else left_line_x
        )
        exit_y = (
            crossing_y_at_line(
                previous_center_x,
                previous_box[3],
                center_x,
                y2,
                exit_line_x,
            )
            if previous_center_x is not None and previous_box is not None
            else None
        )
        is_duplicate = is_recent_duplicate_crossing(
            source,
            direction,
            (x1, y1, x2, y2),
            window_frames=(
                MAX_OCCLUSION_GAP_FRAMES
                if inferred_missing_first_gate
                else RECENT_CROSSING_WINDOW_FRAMES
            ),
            start_frame=crossing_start[0] if crossing_start is not None else None,
            start_y=crossing_start[3] if crossing_start is not None else None,
            exit_y=exit_y,
            person_height=y2 - y1,
        )
        if is_duplicate:
            if track_id in crossing_ids:
                source.pending_directions.pop(track_id, None)
            if debug_counting:
                preserved_ids = [
                    candidate_id
                    for candidate_id in crossing_ids
                    if candidate_id != track_id
                ]
                print(
                    f"\n[计数诊断] {source.name} ID:{track_id} "
                    f"疑似重复跨线，已忽略；保留其他待计ID:{preserved_ids}"
                )
            continue
        for completed_id in crossing_ids:
            source.pending_directions.pop(completed_id, None)
            source.counted_track_directions.add((completed_id, direction))
        completed_track_ids = set(crossing_ids)
        completed_track_ids.add(track_id)
        nested_duplicates = set()
        while True:
            linked_pending_ids = {
                other_id
                for pair, (consecutive_frames, _) in source.nested_track_pairs.items()
                if consecutive_frames >= 3
                and completed_track_ids.intersection(pair)
                for other_id in pair
                if other_id not in completed_track_ids
                and source.pending_directions.get(other_id) == direction
            }
            if not linked_pending_ids:
                break
            nested_duplicates.update(linked_pending_ids)
            completed_track_ids.update(linked_pending_ids)
        for other_id in nested_duplicates:
            source.pending_directions.pop(other_id, None)
            source.counted_track_directions.add((other_id, direction))
        if debug_counting and nested_duplicates:
            print(
                f"\n[计数诊断] {source.name} ID:{track_id} "
                f"连续包含的同向待计轨迹{sorted(nested_duplicates)}已合并 "
                f"{source.debug_frame_context}"
            )
        source.recent_gate_starts = [
            start
            for start in source.recent_gate_starts
            if start[2] not in completed_track_ids
        ]
        source.counted_track_directions.add((track_id, direction))

        if direction == source.entry_direction:
            source.count_in += 1
        else:
            source.count_out += 1
        completed.append((direction, [track_id]))

    source.last_center_x[track_id] = center_x
    source.last_track_boxes[track_id] = (x1, y1, x2, y2, source.frame_index)
    return completed or None


def total_counts(sources):
    total_in = sum(source.count_in for source in sources)
    total_out = sum(source.count_out for source in sources)
    total_current = max(0, total_in - total_out)
    return total_in, total_out, total_current


def parse_source(source):
    return int(source) if source.isdigit() else source


def parse_source_lines(value):
    try:
        left_line, right_line = (int(coordinate.strip()) for coordinate in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "line positions must be two comma-separated integers, such as 150,450"
        ) from error
    if left_line < 0 or right_line < 0 or left_line >= right_line:
        raise argparse.ArgumentTypeError(
            "line positions must satisfy 0 <= left < right"
        )
    return left_line, right_line


def make_line_mouse_callback(source):
    def on_mouse(event, x, _y, flags, _parameter):
        _ = (_y, _parameter)
        if event == cv2.EVENT_LBUTTONDOWN:
            candidates = []
            if source.left_line is not None:
                candidates.append(("left", abs(x - source.left_line)))
            if source.right_line is not None:
                candidates.append(("right", abs(x - source.right_line)))
            if candidates:
                line, distance = min(candidates, key=lambda candidate: candidate[1])
                if distance <= LINE_DRAG_MARGIN:
                    source.dragging_line = line
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            max_x = source.frame_width - 1 if source.frame_width is not None else x
            if source.dragging_line == "left":
                source.left_line = min(max(0, x), max_x)
            elif source.dragging_line == "right":
                source.right_line = min(max(0, x), max_x)
            source.setup_message = ""
        elif event == cv2.EVENT_LBUTTONUP and source.dragging_line is not None:
            source.dragging_line = None

    return on_mouse


def make_gate_mouse_callback(source):
    def on_mouse(event, x, y, flags, parameter):
        _ = (flags, parameter)
        if event != cv2.EVENT_LBUTTONDOWN or source.gate_setup_target is None:
            return
        line_x = (
            source.left_line
            if source.gate_setup_target == "left"
            else source.right_line
        )
        if line_x is None or abs(x - line_x) > LINE_DRAG_MARGIN:
            source.setup_message = "请在线条附近点击通道上、下边界"
            return

        points = source.gate_click_points[source.gate_setup_target]
        if len(points) < 2:
            points.append(min(max(0, y), source.frame_height - 1))
            source.setup_message = ""
        if len(points) == 2:
            gate_y = tuple(sorted(points))
            if gate_y[1] - gate_y[0] < MIN_GATE_LENGTH:
                source.setup_message = (
                    f"门线长度至少为 {MIN_GATE_LENGTH} 像素，请重新选择"
                )
                points.clear()
                return
            if source.gate_setup_target == "left":
                source.left_gate_y = gate_y
            else:
                source.right_gate_y = gate_y
            source.gate_setup_target = None

    return on_mouse


def draw_source_lines(frame, source):
    if source.left_line is not None:
        y1, y2 = source.left_gate_y or (0, frame.shape[0] - 1)
        cv2.line(
            frame,
            (source.left_line, y1),
            (source.left_line, y2),
            (255, 255, 255),
            LINE_OUTLINE_WIDTH,
        )
        cv2.line(
            frame,
            (source.left_line, y1),
            (source.left_line, y2),
            (0, 255, 0),
            LINE_WIDTH,
        )
        draw_line_label(
            frame, source.left_line, "GREEN", (0, 160, 0), source.left_gate_y
        )
    if source.right_line is not None:
        y1, y2 = source.right_gate_y or (0, frame.shape[0] - 1)
        cv2.line(
            frame,
            (source.right_line, y1),
            (source.right_line, y2),
            (255, 255, 255),
            LINE_OUTLINE_WIDTH,
        )
        cv2.line(
            frame,
            (source.right_line, y1),
            (source.right_line, y2),
            (0, 0, 255),
            LINE_WIDTH,
        )
        draw_line_label(
            frame, source.right_line, "RED", (0, 0, 180), source.right_gate_y
        )
    return frame


def draw_gate_setup_points(frame, source):
    for line_name, color in (("left", (0, 255, 0)), ("right", (0, 0, 255))):
        line_x = source.left_line if line_name == "left" else source.right_line
        for y in source.gate_click_points[line_name]:
            if line_x is not None:
                cv2.circle(frame, (line_x, y), 8, color, cv2.FILLED)
                cv2.circle(frame, (line_x, y), 11, (255, 255, 255), 2)
    return frame


def draw_line_label(frame, line_x, label, color, gate_y=None):
    text = f"{label} x={line_x}"
    if gate_y is not None:
        text += f" y={gate_y[0]}-{gate_y[1]}"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    label_x = 8
    label_y = 8 if label == "GREEN" else 8 + text_height + baseline + 12
    cv2.rectangle(
        frame,
        (label_x, label_y),
        (label_x + text_width + 10, label_y + text_height + baseline + 8),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (label_x + 5, label_y + text_height + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def line_instruction():
    return "绿线为左边界，红线为右边界；两条线都已显示，可分别拖动"


def validate_line_positions(source):
    if source.frame_width is None:
        return "无法确认画面宽度，请重新打开视频"
    if source.left_line is None or source.right_line is None:
        return "绿线或红线尚未初始化，请重新打开该路视频"
    if not 0 <= source.left_line < source.frame_width:
        return f"绿线超出画面范围（x={source.left_line}），请重新调整"
    if not 0 <= source.right_line < source.frame_width:
        return f"红线超出画面范围（x={source.right_line}），请重新调整"
    if source.left_line >= source.right_line:
        return (
            f"位置错误：绿线 x={source.left_line} 必须在红线 "
            f"x={source.right_line} 左侧，请拖动后重试"
        )
    if source.right_line - source.left_line < MIN_LINE_GAP:
        return f"两条线间距不足 {MIN_LINE_GAP} 像素，请拉开后重试"
    return None


def validate_gate_positions(source):
    if source.frame_height is None:
        return "无法确认画面高度，请重新打开视频"
    for label, gate_y in (("绿线", source.left_gate_y), ("红线", source.right_gate_y)):
        if gate_y is None:
            return f"{label}门线尚未设置，请按 G/R 设置通道上下边界"
        top, bottom = gate_y
        if not (0 <= top < bottom < source.frame_height):
            return f"{label}门线端点超出画面范围，请重新设置"
        if bottom - top < MIN_GATE_LENGTH:
            return f"{label}门线长度不足 {MIN_GATE_LENGTH} 像素，请重新设置"
    return None


def draw_setup_instructions(frame, source, setup_index, setup_count, font):
    height = frame.shape[0]
    line_positions = (
        f"绿线 x={source.left_line}（左）；红线 x={source.right_line}（右）"
    )
    if setup_index + 1 < setup_count:
        confirm_instruction = "调整好后按空格确认并设置下一路；按 Q 退出"
    else:
        confirm_instruction = "调整好后按空格确认并开始计数；按 Q 退出"

    instructions = (
        f"当前路：{source.name}；{line_positions}",
        "两条线已同时显示；按住线条附近拖动。空格检查位置，Q 退出",
        confirm_instruction,
    )
    for line_index, text in enumerate(instructions):
        position = (20, max(5, height - 100 + line_index * 30))
        frame = draw_chinese_text(frame, text, position, font)
    if source.setup_message:
        frame = draw_chinese_text(
            frame,
            source.setup_message,
            (20, max(5, height - 130)),
            font,
        )
    return frame


def draw_gate_setup_instructions(frame, source, setup_index, setup_count, font):
    height = frame.shape[0]
    if setup_index + 1 < setup_count:
        confirm_instruction = "按空格确认并设置下一路；按 Q 退出"
    else:
        confirm_instruction = "按空格确认门线长度并开始计数；按 Q 退出"
    target_name = (
        "绿线"
        if source.gate_setup_target == "left"
        else "红线"
        if source.gate_setup_target == "right"
        else "未选择"
    )
    instructions = (
        f"当前路：{source.name}；正在设置：{target_name}",
        "按 G 设置绿线、按 R 设置红线；在线条附近点击通道上、下边界各一次",
        "C 清除所有门线范围；Backspace 撤销当前设置的上一个点",
        confirm_instruction,
    )
    for line_index, text in enumerate(instructions):
        frame = draw_chinese_text(
            frame, text, (20, max(5, height - 130 + line_index * 28)), font
        )
    if source.setup_message:
        frame = draw_chinese_text(
            frame, source.setup_message, (20, max(5, height - 160)), font
        )
    return frame


def natural_sort_key(path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def source_video_files(source):
    source_path = Path(source)
    if source_path.is_dir():
        extensions = {
            ".avi",
            ".flv",
            ".m2ts",
            ".m4v",
            ".mkv",
            ".mov",
            ".mp4",
            ".mts",
            ".ts",
            ".wmv",
        }
        return sorted(
            (
                path
                for path in source_path.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            ),
            key=natural_sort_key,
        )
    if source_path.is_file():
        return [source_path]
    return []


def read_source_frame(source):
    while True:
        ret, frame = source.capture.read()
        if ret and frame is not None:
            return True, frame

        source.capture.release()
        source.video_index += 1
        if source.video_index >= len(source.video_files):
            return False, None

        next_file = source.video_files[source.video_index]
        source.capture = cv2.VideoCapture(str(next_file))
        if not source.capture.isOpened():
            source.capture.release()
            raise RuntimeError(f"无法打开视频文件: {next_file}")
        source.fps = video_frame_rate(source.capture)
        source.last_center_x.clear()
        source.last_track_boxes.clear()
        source.track_history.clear()
        source.pending_directions.clear()
        source.counted_track_directions.clear()
        source.recent_crossings.clear()
        source.recent_gate_starts.clear()
        source.nested_track_pairs.clear()
        source.stationary_tracks.clear()
        source.stationary_alerted_ids.clear()
        source.frame_index = 0


def load_chinese_font(size):
    font_paths = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    )
    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    raise FileNotFoundError("未找到可用的中文字体，请安装系统中文字体后重试")


def draw_chinese_text(frame, text, position, font):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(image)
    text_bounds = draw.textbbox(position, text, font=font, stroke_width=2)
    padding = 5
    draw.rectangle(
        (
            text_bounds[0] - padding,
            text_bounds[1] - padding,
            text_bounds[2] + padding,
            text_bounds[3] + padding,
        ),
        fill=(0, 0, 0),
    )
    draw.text(position, text, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_count_overlay(frame, source, sources):
    total_in, total_out, total_current = total_counts(sources)
    source_current = max(0, source.count_in - source.count_out)
    lines = (
        f"{source.window_name}  IN: {source.count_in}  OUT: {source.count_out}  "
        f"CURRENT: {source_current}",
        f"TOTAL ({len(sources)} VIDEOS)  IN: {total_in}  OUT: {total_out}  "
        f"CURRENT: {total_current}",
    )
    height, width = frame.shape[:2]
    scale = min(0.8, max(0.4, (width - 40) / 900))
    thickness = max(1, round(scale * 2))
    panel_height = min(height, max(64, round(scale * 90)))
    panel_y = height - panel_height
    cv2.rectangle(frame, (0, panel_y), (width - 1, height - 1), (0, 0, 0), cv2.FILLED)

    first_y = panel_y + round(panel_height * 0.42)
    second_y = panel_y + round(panel_height * 0.84)
    for line, baseline in zip(lines, (first_y, second_y)):
        cv2.putText(
            frame,
            line,
            (12, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="YOLO people counter with two vertical lines",
        epilog=(
            'Example: python people_counter_dual_gate.py '
            '--source "H:\\新大洋视频\\20261020（2路视频）\\船坞西3_20261023160032" '
            '--source "H:\\新大洋视频\\20261020（2路视频）\\船坞东2_20261023155926" '
            '--entry-direction left-to-right --entry-direction left-to-right '
            '--source-lines 150,450 --source-lines 220,520'
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        help="camera index or video path; repeat this option to add more sources",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(bundled_resource_path("yolov8n.pt")),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="person detection threshold; 0.1 enables ByteTrack low-confidence recovery",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="inference image size; larger values can improve small/occluded person detection",
    )
    parser.add_argument(
        "--debug-counting",
        action="store_true",
        help="print per-frame track boxes, video positions and crossing diagnostics",
    )
    parser.add_argument(
        "--gate-assist-source",
        action="append",
        type=int,
        metavar="INDEX",
        help=(
            "enable additional gate-area detection for this 1-based source index; "
            "repeat to enable multiple sources (slower, disabled by default)"
        ),
    )
    parser.add_argument(
        "--source-lines",
        action="append",
        type=parse_source_lines,
        metavar="LEFT,RIGHT",
        help="left and right x positions for this source; repeat in --source order",
    )
    parser.add_argument(
        "--entry-direction",
        action="append",
        choices=("left-to-right", "right-to-left"),
        metavar="DIRECTION",
        help=(
            "physical direction counted as entry for this source; repeat in "
            "--source order (left-to-right by default)"
        ),
    )
    parser.add_argument(
        "--left-line",
        type=int,
        help="default left-line x coordinate for every source (otherwise click to set)",
    )
    parser.add_argument(
        "--right-line",
        type=int,
        help="default right-line x coordinate for every source (otherwise click to set)",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.is_file():
        parser.error(
            f"model file not found: {weights_path}\n"
            "Copy the complete PeopleCounter folder to the target computer "
            "and confirm _internal\\yolov8n.pt exists."
        )

    if (args.left_line is not None or args.right_line is not None) and args.source_lines:
        parser.error("--source-lines cannot be combined with --left-line or --right-line")

    source_args = args.source or ["0"]
    gate_assist_sources = set(args.gate_assist_source or [])
    if any(index < 1 or index > len(source_args) for index in gate_assist_sources):
        parser.error("--gate-assist-source must be a 1-based index of a configured source")
    if args.entry_direction and len(args.entry_direction) != len(source_args):
        parser.error(
            f"provide one --entry-direction for each --source "
            f"({len(source_args)} sources, {len(args.entry_direction)} directions)"
        )
    # Both configured gates count left-to-right as entry unless explicitly overridden.
    entry_directions = args.entry_direction or ["left-to-right"] * len(source_args)
    if args.source_lines:
        if len(args.source_lines) != len(source_args):
            parser.error(
                f"provide one --source-lines for each --source "
                f"({len(source_args)} sources, {len(args.source_lines)} line pairs)"
            )
        source_line_pairs = args.source_lines
    elif args.left_line is not None or args.right_line is not None:
        default_left = 150 if args.left_line is None else args.left_line
        default_right = 450 if args.right_line is None else args.right_line
        if default_left < 0 or default_left >= default_right:
            parser.error("--left-line must be smaller than --right-line")
        source_line_pairs = [(default_left, default_right) for _ in source_args]
    else:
        source_line_pairs = [(None, None) for _ in source_args]
    chinese_font = load_chinese_font(24)
    sources = []

    try:
        for index, source_arg in enumerate(source_args, start=1):
            source_value = parse_source(source_arg)
            video_files = source_video_files(source_arg)
            if Path(source_arg).is_dir() and not video_files:
                raise FileNotFoundError(f"文件夹内未找到支持的视频文件: {source_arg}")
            if video_files:
                capture = cv2.VideoCapture(str(video_files[0]))
                source_name = Path(source_arg).name if Path(source_arg).is_dir() else f"Source {index}"
            elif isinstance(source_value, int):
                if platform.system() == "Windows":
                    capture = cv2.VideoCapture(source_value, cv2.CAP_DSHOW)
                else:
                    capture = cv2.VideoCapture(source_value)
                source_name = f"Source {index}"
            else:
                raise FileNotFoundError(f"未找到视频文件或文件夹: {source_arg}")
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(
                    f"无法打开视频源: {video_files[0] if video_files else source_arg}"
                )
            sources.append(
                SourceState(
                    name=source_name,
                    window_name=f"Video {index}",
                    capture=capture,
                    left_line=source_line_pairs[index - 1][0],
                    right_line=source_line_pairs[index - 1][1],
                    fps=video_frame_rate(capture),
                    entry_direction=entry_directions[index - 1],
                    video_files=video_files,
                )
            )

        preview_frames = {}
        setup_cancelled = False
        interactive_sources = [
            source
            for source in sources
            if (
                source.left_line is None
                or source.right_line is None
                or source.left_gate_y is None
                or source.right_gate_y is None
            )
        ]
        for setup_index, source in enumerate(interactive_sources):
            ret, frame = source.capture.read()
            if not ret or frame is None:
                raise RuntimeError(f"无法读取视频首帧: {source.name}")
            source.frame_width = frame.shape[1]
            source.frame_height = frame.shape[0]
            needs_line_setup = (
                source.left_line is None or source.right_line is None
            )
            if source.left_line is None:
                source.left_line = source.frame_width // 3
            if source.right_line is None:
                source.right_line = (source.frame_width * 2) // 3
            if source.frame_width <= MIN_LINE_GAP:
                raise ValueError(
                    f"{source.name}: 视频宽度不足以设置间距至少 "
                    f"{MIN_LINE_GAP} 像素的两条计数线"
                )
            preview_frames[id(source)] = frame
            cv2.namedWindow(source.window_name)
            source.window_created = True
            if needs_line_setup:
                cv2.setMouseCallback(
                    source.window_name, make_line_mouse_callback(source)
                )
                while True:
                    frame = preview_frames[id(source)].copy()
                    draw_source_lines(frame, source)
                    frame = draw_setup_instructions(
                        frame,
                        source,
                        setup_index,
                        len(interactive_sources),
                        chinese_font,
                    )
                    cv2.imshow(source.window_name, frame)

                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("q"):
                        setup_cancelled = True
                        break
                    if key in (10, 13, 32):
                        validation_error = validate_line_positions(source)
                        if validation_error is None:
                            source.setup_message = ""
                            break
                        source.setup_message = validation_error
            else:
                validation_error = validate_line_positions(source)
                if validation_error is not None:
                    raise ValueError(f"{source.name}: {validation_error}")

            if not setup_cancelled and (
                source.left_gate_y is None or source.right_gate_y is None
            ):
                source.gate_setup_target = None
                source.gate_click_points = {"left": [], "right": []}
                cv2.setMouseCallback(
                    source.window_name, make_gate_mouse_callback(source)
                )
                while True:
                    frame = preview_frames[id(source)].copy()
                    draw_source_lines(frame, source)
                    frame = draw_gate_setup_points(frame, source)
                    frame = draw_gate_setup_instructions(
                        frame,
                        source,
                        setup_index,
                        len(interactive_sources),
                        chinese_font,
                    )
                    cv2.imshow(source.window_name, frame)

                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("q"):
                        setup_cancelled = True
                        break
                    if key in (ord("g"), ord("G")):
                        source.gate_setup_target = "left"
                        source.left_gate_y = None
                        source.gate_click_points["left"].clear()
                        source.setup_message = ""
                    elif key in (ord("r"), ord("R")):
                        source.gate_setup_target = "right"
                        source.right_gate_y = None
                        source.gate_click_points["right"].clear()
                        source.setup_message = ""
                    elif key in (ord("c"), ord("C")):
                        source.left_gate_y = None
                        source.right_gate_y = None
                        source.gate_click_points = {"left": [], "right": []}
                        source.gate_setup_target = None
                        source.setup_message = ""
                    elif key in (8, 127):
                        if (
                            source.gate_setup_target is not None
                            and source.gate_click_points[source.gate_setup_target]
                        ):
                            source.gate_click_points[
                                source.gate_setup_target
                            ].pop()
                        source.setup_message = ""
                    elif key in (10, 13, 32):
                        validation_error = validate_gate_positions(source)
                        if validation_error is None:
                            source.setup_message = ""
                            break
                        source.setup_message = validation_error

            cv2.destroyWindow(source.window_name)
            source.window_created = False
            if setup_cancelled:
                break

        started = not setup_cancelled
        if started:
            print("计数线及门线长度设置已确认，正在加载两路检测模型……")
            for index, source in enumerate(sources, start=1):
                source.model = YOLO(str(weights_path))
                if index in gate_assist_sources:
                    add_gate_assist(
                        source.model, YOLO(str(weights_path)), source,
                        args.conf, args.iou, args.imgsz,
                    )
                    print(f"{source.name}: 已启用门线区域辅助识别，处理速度可能降低")

        while started and any(source.active for source in sources):
            for source in sources:
                if not source.active:
                    continue

                preview_frame = preview_frames.pop(id(source), None)
                if preview_frame is None:
                    ret, frame = read_source_frame(source)
                else:
                    ret, frame = True, preview_frame
                if not ret or frame is None:
                    source.active = False
                    continue
                source.frame_index += 1
                if args.debug_counting:
                    source.debug_frame_context = make_debug_frame_context(source)

                left_line = source.left_line
                right_line = source.right_line
                if right_line is None or left_line is None:
                    continue
                width = frame.shape[1]
                source.frame_width = width
                if left_line < 0 or right_line >= width:
                    raise ValueError(
                        f"{source.name}: line coordinates ({left_line}, {right_line}) "
                        f"are outside "
                        f"the video width ({width}); adjust --source-lines"
                    )

                if source.model is None:
                    raise RuntimeError(f"{source.name}: 检测模型未加载")
                results = source.model.track(
                    frame,
                    persist=True,
                    tracker=str(Path(__file__).with_name("people_counter_bytetrack.yaml")),
                    classes=[0],
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    verbose=False,
                )

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes.xyxy) > 0:
                        ids = boxes.id.int().tolist() if boxes.id is not None else []
                        coordinates = boxes.xyxy.cpu().numpy()
                        update_nested_track_pairs(source, ids, coordinates)
                        if args.debug_counting:
                            log_debug_track_boxes(source, ids, coordinates)

                        for index, box in enumerate(coordinates):
                            x1, y1, x2, y2 = map(int, box)
                            track_id = ids[index] if index < len(ids) else None

                            if track_id is not None:
                                completed = update_track_crossing(
                                    source,
                                    track_id,
                                    (x1, y1, x2, y2),
                                    left_line,
                                    right_line,
                                    debug_counting=args.debug_counting,
                                )
                                if completed is not None:
                                    for direction, completed_ids in completed:
                                        crossing_label = "进入" if direction == source.entry_direction else "离开"
                                        total_in = sum(item.count_in for item in sources)
                                        total_out = sum(item.count_out for item in sources)
                                        print(
                                            f"\n{source.name} "
                                            f"IDs:{completed_ids} {crossing_label} "
                                            f"{len(completed_ids)}人 | "
                                            f"本路进入:{source.count_in} "
                                            f"离开:{source.count_out} | "
                                            f"两路合计进入:{total_in} "
                                            f"离开:{total_out}"
                                            + (
                                                f" | {source.debug_frame_context}"
                                                if args.debug_counting
                                                else ""
                                            )
                                        )

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            label = f"ID:{track_id}" if track_id is not None else "Person"
                            cv2.putText(
                                frame,
                                label,
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2,
                            )
                    else:
                        update_nested_track_pairs(source, [], [])
                else:
                    update_nested_track_pairs(source, [], [])

                draw_source_lines(frame, source)
                total_in, total_out, total_current = total_counts(sources)
                frame = draw_count_overlay(frame, source, sources)
                if not source.window_created:
                    cv2.namedWindow(source.window_name)
                    cv2.setMouseCallback(
                        source.window_name, make_line_mouse_callback(source)
                    )
                    source.window_created = True
                cv2.imshow(source.window_name, frame)
                print(
                    f"\r总计（{len(sources)}路）| 进入: {total_in}  离开: {total_out}  "
                    f"当前人数: {total_current}",
                    end="",
                    flush=True,
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    started = False
                    break
    finally:
        if sources:
            print()
        for source in sources:
            source.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()