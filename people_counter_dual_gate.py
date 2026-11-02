import argparse
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MAX_OCCLUSION_GAP_FRAMES = 180
RECENT_CROSSING_WINDOW_FRAMES = 180
LINE_DRAG_MARGIN = 30
LINE_OUTLINE_WIDTH = 8
LINE_WIDTH = 4
MIN_LINE_GAP = 30


def bundled_resource_path(filename):
    bundle_dir = getattr(sys, "_MEIPASS", None)
    base_dir = Path(bundle_dir) if bundle_dir else Path(__file__).resolve().parent
    return base_dir / filename


@dataclass
class SourceState:
    name: str
    window_name: str
    capture: cv2.VideoCapture
    model: YOLO | None = None
    left_line: int | None = None
    right_line: int | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    dragging_line: str | None = None
    setup_message: str = ""
    window_created: bool = False
    video_files: list = field(default_factory=list)
    video_index: int = 0
    last_center_x: dict = field(default_factory=dict)
    last_track_boxes: dict = field(default_factory=dict)
    pending_directions: dict = field(default_factory=dict)
    pending_gate_positions: dict = field(default_factory=dict)
    recent_crossings: list = field(default_factory=list)
    roi_shape: str | None = None
    roi_points: list[tuple[int, int]] | None = None
    roi_click_points: list[tuple[int, int]] = field(default_factory=list)
    frame_index: int = 0
    entry_direction: str = "left-to-right"
    count_in: int = 0
    count_out: int = 0
    active: bool = True


def crossing_y_at_line(previous_x, previous_y, current_x, current_y, line_x):
    delta_x = current_x - previous_x
    if delta_x == 0:
        return None
    ratio = (line_x - previous_x) / delta_x
    if not 0 <= ratio <= 1:
        return None
    return line_x, previous_y + ratio * (current_y - previous_y)


def point_in_roi(source, point, tolerance=0):
    if source.roi_points is None:
        return False
    contour = np.asarray(source.roi_points, dtype=np.int32)
    return cv2.pointPolygonTest(
        contour, (float(point[0]), float(point[1])), True
    ) >= -tolerance


def roi_crossing_tolerance(*boxes):
    largest_box_height = max(box[3] - box[1] for box in boxes)
    return max(6, largest_box_height * 0.15)


def crossing_is_inside_roi(source, direction, previous_box, current_box, left_line_x, right_line_x):
    previous_x = (previous_box[0] + previous_box[2]) / 2
    previous_y = previous_box[3]
    current_x = (current_box[0] + current_box[2]) / 2
    current_y = current_box[3]
    first_line_x, second_line_x = (
        (left_line_x, right_line_x)
        if direction == "left-to-right"
        else (right_line_x, left_line_x)
    )
    crossing_points = (
        crossing_y_at_line(previous_x, previous_y, current_x, current_y, first_line_x),
        crossing_y_at_line(previous_x, previous_y, current_x, current_y, second_line_x),
    )
    tolerance = roi_crossing_tolerance(previous_box, current_box)
    return all(
        point is not None and point_in_roi(source, point, tolerance)
        for point in crossing_points
    )


def build_roi_points(shape, clicks):
    if shape == "rectangle" and len(clicks) == 2:
        (x1, y1), (x2, y2) = clicks
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return [(left, top), (right, top), (right, bottom), (left, bottom)]
    if shape == "parallelogram" and len(clicks) == 3:
        top_left, top_right, bottom_right = clicks
        bottom_left = (
            top_left[0] + bottom_right[0] - top_right[0],
            top_left[1] + bottom_right[1] - top_right[1],
        )
        return [top_left, top_right, bottom_right, bottom_left]
    return None


def validate_roi_positions(source, points):
    if points is None or len(points) != 4:
        return "请按提示完成区域顶点设置"
    if source.frame_width is None:
        return "无法确认画面宽度，请重新打开视频"
    height = source.frame_height
    if height is None:
        return "无法确认画面高度，请重新打开视频"
    if any(not (0 <= x < source.frame_width and 0 <= y < height) for x, y in points):
        return "区域顶点超出画面范围，请重新设置"
    contour = np.asarray(points, dtype=np.int32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 1000:
        return "通道区域形状无效或过小，请重新设置顶点"
    if (
        source.left_line is None
        or source.right_line is None
        or not min(x for x, _ in points) < source.left_line
        or not max(x for x, _ in points) > source.right_line
    ):
        return "通道区域必须同时覆盖绿线和红线"
    return None


def make_roi_mouse_callback(source):
    def on_mouse(event, x, y, flags, parameter):
        _ = (flags, parameter)
        if event != cv2.EVENT_LBUTTONDOWN or source.roi_shape is None:
            return
        required_clicks = 2 if source.roi_shape == "rectangle" else 3
        if len(source.roi_click_points) < required_clicks:
            source.roi_click_points.append(
                (
                    min(max(0, x), source.frame_width - 1),
                    min(max(0, y), source.frame_height - 1),
                )
            )
            source.setup_message = ""

    return on_mouse


def gate_transition(previous_center_x, current_center_x, left_line_x, right_line_x):
    if previous_center_x < left_line_x:
        if current_center_x >= right_line_x:
            return "through", "left-to-right"
        if current_center_x >= left_line_x:
            return "start", "left-to-right"
    elif previous_center_x > right_line_x:
        if current_center_x <= left_line_x:
            return "through", "right-to-left"
        if current_center_x <= right_line_x:
            return "start", "right-to-left"
    elif left_line_x <= previous_center_x <= right_line_x:
        if current_center_x > right_line_x:
            return "complete", "left-to-right"
        if current_center_x < left_line_x:
            return "complete", "right-to-left"
    return None


def pending_tracks_in_box(
    box, frame_index, pending_directions, last_track_boxes, direction, current_track_id
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


def pending_tracks_across_gate(
    source, box, current_center_x, left_line_x, right_line_x, direction, current_track_id
):
    _, y1, _, y2 = box
    current_center_y = (y1 + y2) / 2
    current_height = y2 - y1
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
        transition = gate_transition(
            previous_center_x, current_center_x, left_line_x, right_line_x
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


def is_recent_duplicate_crossing(source, direction, box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    source.recent_crossings = [
        crossing
        for crossing in source.recent_crossings
        if source.frame_index - crossing[0] <= RECENT_CROSSING_WINDOW_FRAMES
    ]

    for _, previous_direction, previous_box in source.recent_crossings:
        if previous_direction != direction:
            continue
        px1, py1, px2, py2 = previous_box
        previous_center_x = (px1 + px2) / 2
        previous_center_y = (py1 + py2) / 2
        horizontal_tolerance = max(60, ((px2 - px1) + width) * 0.75)
        vertical_tolerance = max(60, ((py2 - py1) + height) * 0.75)
        if (
            abs(previous_center_x - center_x) <= horizontal_tolerance
            and abs(previous_center_y - center_y) <= vertical_tolerance
        ):
            return True

    source.recent_crossings.append(
        (source.frame_index, direction, (x1, y1, x2, y2))
    )
    return False


def update_track_crossing(
    source, track_id, box, left_line_x, right_line_x, debug_counting=False
):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    previous_box = source.last_track_boxes.get(track_id)
    previous_center_x = source.last_center_x.get(track_id)
    if debug_counting and previous_center_x is None:
        print(
            f"\n[计数诊断] {source.name} 首次检测 ID:{track_id} "
            f"x={center_x:.1f}，绿线x={left_line_x}，红线x={right_line_x}"
        )
    expired_ids = [
        pending_id
        for pending_id in source.pending_directions
        if pending_id != track_id
        and (
            pending_id not in source.last_track_boxes
            or source.frame_index - source.last_track_boxes[pending_id][4]
            > MAX_OCCLUSION_GAP_FRAMES
        )
    ]
    for pending_id in expired_ids:
        source.pending_directions.pop(pending_id, None)
        source.pending_gate_positions.pop(pending_id, None)
        source.last_center_x.pop(pending_id, None)
        source.last_track_boxes.pop(pending_id, None)

    transition = None
    if previous_center_x is not None:
        transition = gate_transition(
            previous_center_x, center_x, left_line_x, right_line_x
        )
    if (
        transition is not None
        and transition[0] == "complete"
        and source.pending_directions.get(track_id) not in (None, transition[1])
    ):
        source.pending_directions.pop(track_id, None)
        source.pending_gate_positions.pop(track_id, None)
    if transition is not None and transition[0] == "start":
        source.pending_directions[track_id] = transition[1]
        start_line_x = (
            left_line_x
            if transition[1] == "left-to-right"
            else right_line_x
        )
        if previous_box is not None:
            start_point = crossing_y_at_line(
                previous_center_x,
                previous_box[3],
                center_x,
                y2,
                start_line_x,
            )
            if start_point is not None:
                source.pending_gate_positions[track_id] = start_point
    if debug_counting and transition is not None:
        print(
            f"\n[计数诊断] {source.name} ID:{track_id} "
            f"x={previous_center_x:.1f}->{center_x:.1f} "
            f"状态={transition[0]} 方向={transition[1]}"
        )

    completed = []
    for direction in ("left-to-right", "right-to-left"):
        candidate_ids = pending_tracks_in_box(
            (x1, y1, x2, y2),
            source.frame_index,
            source.pending_directions,
            source.last_track_boxes,
            direction,
            track_id,
        )
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
        ):
            crossing_ids.insert(0, track_id)

        if not crossing_ids:
            crossing_ids = pending_tracks_across_gate(
                source,
                (x1, y1, x2, y2),
                center_x,
                left_line_x,
                right_line_x,
                direction,
                track_id,
            )
        if not crossing_ids:
            continue
        end_line_x = (
            right_line_x if direction == "left-to-right" else left_line_x
        )
        valid_crossing = False
        for candidate_id in crossing_ids:
            if (
                candidate_id == track_id
                and transition is not None
                and transition[0] == "through"
                and previous_box is not None
            ):
                valid_crossing = crossing_is_inside_roi(
                    source,
                    direction,
                    previous_box,
                    (x1, y1, x2, y2),
                    left_line_x,
                    right_line_x,
                )
            else:
                start_point = source.pending_gate_positions.get(candidate_id)
                candidate_box = source.last_track_boxes.get(candidate_id)
                if start_point is None or candidate_box is None:
                    continue
                end_point = crossing_y_at_line(
                    (candidate_box[0] + candidate_box[2]) / 2,
                    candidate_box[3],
                    center_x,
                    y2,
                    end_line_x,
                )
                tolerance = roi_crossing_tolerance(
                    candidate_box, (x1, y1, x2, y2)
                )
                valid_crossing = (
                    end_point is not None
                    and point_in_roi(source, start_point, tolerance)
                    and point_in_roi(source, end_point, tolerance)
                )
            if valid_crossing:
                break

        for completed_id in crossing_ids:
            source.pending_directions.pop(completed_id, None)
            source.pending_gate_positions.pop(completed_id, None)
        if not valid_crossing:
            if debug_counting:
                print(
                    f"\n[计数诊断] {source.name} ID:{track_id} "
                    f"未完整通过通道区域，忽略"
                )
            continue

        is_duplicate = is_recent_duplicate_crossing(
            source, direction, (x1, y1, x2, y2)
        )
        if is_duplicate:
            if debug_counting:
                print(
                    f"\n[计数诊断] {source.name} ID:{track_id} "
                    f"疑似重复跨线，已忽略"
                )
            continue
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


def draw_source_lines(frame, source):
    height = frame.shape[0]
    if source.roi_points:
        overlay = frame.copy()
        contour = np.asarray(source.roi_points, dtype=np.int32)
        cv2.fillPoly(overlay, [contour], (255, 180, 0))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [contour], True, (255, 180, 0), 3)
    if source.left_line is not None:
        cv2.line(
            frame,
            (source.left_line, 0),
            (source.left_line, height),
            (255, 255, 255),
            LINE_OUTLINE_WIDTH,
        )
        cv2.line(
            frame,
            (source.left_line, 0),
            (source.left_line, height),
            (0, 255, 0),
            LINE_WIDTH,
        )
        draw_line_label(frame, source.left_line, "GREEN", (0, 160, 0))
    if source.right_line is not None:
        cv2.line(
            frame,
            (source.right_line, 0),
            (source.right_line, height),
            (255, 255, 255),
            LINE_OUTLINE_WIDTH,
        )
        cv2.line(
            frame,
            (source.right_line, 0),
            (source.right_line, height),
            (0, 0, 255),
            LINE_WIDTH,
        )
        draw_line_label(frame, source.right_line, "RED", (0, 0, 180))
    return frame


def draw_roi_setup(frame, source):
    points = source.roi_points or build_roi_points(
        source.roi_shape, source.roi_click_points
    )
    if points:
        contour = np.asarray(points, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [contour], (255, 180, 0))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [contour], True, (255, 180, 0), 3)
    elif source.roi_click_points:
        for point in source.roi_click_points:
            cv2.circle(frame, point, 7, (255, 180, 0), cv2.FILLED)
        if len(source.roi_click_points) > 1:
            cv2.polylines(
                frame,
                [np.asarray(source.roi_click_points, dtype=np.int32)],
                False,
                (255, 180, 0),
                2,
            )
    for point in source.roi_click_points:
        cv2.circle(frame, point, 7, (255, 255, 255), 2)
    return frame


def draw_line_label(frame, line_x, label, color):
    text = f"{label} x={line_x}"
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


def draw_roi_setup_instructions(frame, source, setup_index, setup_count, font):
    height = frame.shape[0]
    shape_name = (
        "矩形"
        if source.roi_shape == "rectangle"
        else "平行四边形"
        if source.roi_shape == "parallelogram"
        else "未选择"
    )
    if setup_index + 1 < setup_count:
        confirm_instruction = "按空格确认并设置下一路；按 Q 退出"
    else:
        confirm_instruction = "按空格确认通道区域并开始计数；按 Q 退出"
    instructions = (
        f"当前路：{source.name}；通道区域：{shape_name}",
        "按 R 后点击矩形对角顶点；按 P 后顺时针点击平行四边形前三个顶点",
        "C 清除重画；Backspace 撤销上一个点；区域必须覆盖绿线和红线",
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
        source.last_center_x.clear()
        source.last_track_boxes.clear()
        source.pending_directions.clear()
        source.pending_gate_positions.clear()
        source.recent_crossings.clear()
        source.frame_index = 0


def load_chinese_font(size):
    font_paths = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    )
    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    raise FileNotFoundError("未找到 Windows 中文字体（微软雅黑、黑体或宋体）")


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
        default=960,
        help="inference image size; 960 is the balanced local tracking test setting",
    )
    parser.add_argument(
        "--debug-counting",
        action="store_true",
        help="print detected track positions and line-crossing transitions",
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
            elif isinstance(source_value, int) and platform.system() == "Windows":
                capture = cv2.VideoCapture(source_value, cv2.CAP_DSHOW)
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
                or source.roi_points is None
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

            if not setup_cancelled and source.roi_points is None:
                source.roi_shape = None
                source.roi_click_points.clear()
                cv2.setMouseCallback(
                    source.window_name, make_roi_mouse_callback(source)
                )
                while True:
                    frame = preview_frames[id(source)].copy()
                    draw_source_lines(frame, source)
                    frame = draw_roi_setup(frame, source)
                    frame = draw_roi_setup_instructions(
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
                    if key in (ord("r"), ord("R")):
                        source.roi_shape = "rectangle"
                        source.roi_click_points.clear()
                        source.setup_message = ""
                    elif key in (ord("p"), ord("P")):
                        source.roi_shape = "parallelogram"
                        source.roi_click_points.clear()
                        source.setup_message = ""
                    elif key in (ord("c"), ord("C")):
                        source.roi_click_points.clear()
                        source.roi_points = None
                        source.setup_message = ""
                    elif key in (8, 127):
                        if source.roi_click_points:
                            source.roi_click_points.pop()
                        source.setup_message = ""
                    elif key in (10, 13, 32):
                        candidate_points = build_roi_points(
                            source.roi_shape, source.roi_click_points
                        )
                        validation_error = validate_roi_positions(
                            source, candidate_points
                        )
                        if validation_error is None:
                            source.roi_points = candidate_points
                            source.setup_message = ""
                            break
                        source.setup_message = validation_error

            cv2.destroyWindow(source.window_name)
            source.window_created = False
            if setup_cancelled:
                break

        started = not setup_cancelled
        if started:
            print("线位及通道区域设置已确认，正在加载两路检测模型……")
            for source in sources:
                source.model = YOLO(str(weights_path))

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