from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import torch
import unittest
from ultralytics.engine.results import Results

from people_counter_dual_gate import (
    SourceState,
    add_gate_assist,
    gate_assist_crop,
    log_debug_track_boxes,
    make_debug_frame_context,
    merge_gate_assist_detections,
    total_counts,
    update_nested_track_pairs,
    update_track_crossing,
)


class GateAssistTests(unittest.TestCase):
    def test_crop_includes_both_gates_and_is_clipped_to_frame(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        crop, left = gate_assist_crop(frame, 835, 1005)
        self.assertEqual(left, 648)
        self.assertEqual(crop.shape, (1080, 544, 3))
        crop, left = gate_assist_crop(frame, 20, 200)
        self.assertEqual(left, 0)
        self.assertEqual(crop.shape[0], 1080)

    def test_merge_restores_coordinates_and_suppresses_overlap(self):
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        result = Results(frame, path="test", names={0: "person"}, boxes=torch.tensor([
            [120, 40, 180, 160, 0.9, 0],
        ]))
        crop_result = Results(frame[:, 100:400], path="crop", names={0: "person"},
            boxes=torch.tensor([
                [21, 40, 81, 160, 0.8, 0],
                [150, 50, 200, 180, 0.7, 0],
                [0, 30, 30, 100, 0.6, 0],
            ]))
        merge_gate_assist_detections(result, crop_result, 100, 300, 0.5)
        self.assertEqual(len(result.boxes), 2)
        self.assertEqual(result.boxes.xyxy.tolist(), [
            [120, 40, 180, 160], [250, 50, 300, 180],
        ])

    def test_assist_callback_runs_before_tracker_and_preserves_full_frame(self):
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        result = Results(frame, path="test", names={0: "person"},
                         boxes=torch.empty((0, 6)))
        crop_result = Results(frame[:, :328], path="crop", names={0: "person"},
                              boxes=torch.tensor([[30, 40, 90, 160, 0.8, 0]]))
        model = Mock()
        assist_model = Mock()
        assist_model.predict.return_value = [crop_result]
        source = SimpleNamespace(left_line=100, right_line=200, name="test")
        add_gate_assist(model, assist_model, source, 0.1, 0.5, 1280)
        event, callback = model.add_callback.call_args.args
        self.assertEqual(event, "on_predict_postprocess_end")
        callback(SimpleNamespace(results=[result], args=SimpleNamespace(device="cpu")))
        self.assertEqual(result.boxes.xyxy.tolist(), [[30, 40, 90, 160]])
        self.assertIs(result.orig_img, frame)


class GateCrossingTests(unittest.TestCase):
    def setUp(self):
        self.source = SourceState(
            name="test",
            window_name="test",
            capture=cv2.VideoCapture(),
            left_line=100,
            right_line=200,
            frame_width=320,
            frame_height=320,
            left_gate_y=(0, 319),
            right_gate_y=(0, 319),
        )

    def observe(self, frame_index, track_id, center_x, center_y=100):
        self.source.frame_index = frame_index
        return update_track_crossing(
            self.source,
            track_id,
            (center_x - 10, center_y - 20, center_x + 10, center_y + 20),
            100,
            200,
        )

    def observe_box(self, frame_index, track_id, box):
        self.source.frame_index = frame_index
        return update_track_crossing(
            self.source,
            track_id,
            box,
            100,
            200,
        )

    def tearDown(self):
        self.source.capture.release()

    def test_debug_context_uses_current_video_and_backend_position(self):
        capture = Mock()
        capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_POS_FRAMES: 126,
            cv2.CAP_PROP_POS_MSEC: 123456,
        }[prop]
        source = SimpleNamespace(
            capture=capture,
            video_files=[Path("first.mp4"), Path("second.mp4")],
            video_index=1,
            frame_index=126,
        )

        context = make_debug_frame_context(source)

        self.assertIn("处理时间=", context)
        self.assertIn("视频文件=second.mp4", context)
        self.assertIn("文件帧号（从0开始）=125", context)
        self.assertIn("解码器报告视频位置=00:02:03.456", context)
        self.assertIn("帧=126", context)

    def test_debug_context_marks_unavailable_video_position_and_live_source(self):
        capture = Mock()
        capture.get.side_effect = lambda prop: (
            0 if prop == cv2.CAP_PROP_POS_FRAMES else float("nan")
        )
        source = SimpleNamespace(
            capture=capture, video_files=[Path("video.mp4")],
            video_index=0, frame_index=5,
        )
        context = make_debug_frame_context(source)
        self.assertIn("文件帧号（从0开始）=不可用", context)
        self.assertIn("解码器报告视频位置=不可用", context)

        source.video_files = []
        self.assertIn("视频位置=不适用（实时源）", make_debug_frame_context(source))

    def test_debug_logs_all_simultaneous_track_boxes_and_crossings(self):
        self.source.debug_frame_context = "视频文件=example.mp4 帧=2"
        self.source.frame_index = 1
        output = StringIO()
        with redirect_stdout(output):
            log_debug_track_boxes(
                self.source,
                [44, 49],
                [(40, 50, 60, 90), (42, 30, 62, 120)],
            )
            update_track_crossing(
                self.source, 44, (40, 50, 60, 90), 100, 200,
                debug_counting=True,
            )
            self.source.frame_index = 2
            update_track_crossing(
                self.source, 44, (110, 50, 130, 90), 100, 200,
                debug_counting=True,
            )
        text = output.getvalue()
        self.assertIn("ID:44框=(40,50,60,90)", text)
        self.assertIn("ID:49框=(42,30,62,120)", text)
        self.assertIn("首次检测 ID:44", text)
        self.assertIn("状态=start", text)
        self.assertIn("框=(110,50,130,90)", text)
        self.assertEqual(text.count("视频文件=example.mp4 帧=2"), 3)

    def test_total_current_is_clamped_when_exits_exceed_entries(self):
        self.source.count_in = 2
        self.source.count_out = 3

        self.assertEqual(total_counts([self.source]), (2, 3, 0))

    def test_total_current_uses_aggregate_entry_and_exit_counts(self):
        second_source = SourceState(
            name="second",
            window_name="second",
            capture=cv2.VideoCapture(),
            count_in=3,
            count_out=1,
        )
        self.source.count_in = 2
        self.source.count_out = 3
        try:
            self.assertEqual(
                total_counts([self.source, second_source]),
                (5, 4, 1),
            )
        finally:
            second_source.capture.release()

    def test_counts_full_right_to_left_crossing(self):
        self.observe(1, 1, 250)
        self.observe(2, 1, 180)
        self.observe(3, 1, 80)

        self.assertEqual(self.source.count_out, 1)

    def test_counts_full_left_to_right_crossing(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 120)
        self.observe(3, 1, 250)

        self.assertEqual(self.source.count_in, 1)

    def test_counts_track_completing_crossing_after_first_seen_inside_gates(self):
        self.observe(1, 1, 150)
        self.observe(2, 1, 80)

        self.assertEqual(self.source.count_out, 1)

    def test_missing_first_gate_state_fallback_does_not_count_same_track_twice(self):
        self.observe(1, 1, 150)
        self.observe(2, 1, 80)
        self.observe(3, 1, 150)
        self.observe(4, 1, 80)

        self.assertEqual(self.source.count_out, 1)

    def test_missing_gate_fallback_suppresses_recent_same_path_count_after_id_switch(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)
        self.observe(3, 1, 250)
        self.observe(30, 54, 190)
        self.observe(31, 54, 210)

        self.assertEqual(self.source.count_in, 1)

    def test_missing_gate_fallback_does_not_duplicate_active_first_gate_track(self):
        self.observe(1, 35, 50)
        self.observe(2, 35, 150)
        self.observe(3, 35, 250)
        self.observe(4, 50, 90, center_y=100)
        self.observe(5, 50, 110, center_y=100)
        self.observe(6, 54, 190, center_y=300)
        self.observe(7, 54, 210, center_y=300)

        self.assertEqual(self.source.count_in, 1)

    def test_missing_gate_fallback_uses_recent_start_after_pending_state_is_lost(self):
        self.observe(1, 50, 90, center_y=90)
        self.observe(2, 50, 110, center_y=110)
        self.source.pending_directions.pop(50)

        self.observe(3, 54, 180, center_y=230)
        self.observe(4, 54, 195, center_y=245)
        self.observe(5, 54, 205, center_y=255)

        self.assertEqual(self.source.count_in, 1)

    def test_reassociates_start_and_completion_after_track_switch_within_ten_seconds(self):
        self.observe(1, 26, 90)
        self.observe(2, 26, 110)
        self.observe(402, 29, 190)
        self.observe(403, 29, 210)

        self.assertEqual(self.source.count_in, 1)
        self.assertNotIn(26, self.source.pending_directions)

    def test_discards_uncompleted_first_gate_state_after_ten_seconds(self):
        self.observe(1, 26, 90)
        self.observe(2, 26, 110)

        self.observe(503, 29, 150)

        self.assertNotIn(26, self.source.pending_directions)
        self.assertEqual(self.source.count_in, 0)

    def test_stationary_person_in_channel_triggers_one_alarm_after_thirty_seconds(self):
        output = StringIO()
        with redirect_stdout(output):
            for frame_index in range(1, 1503):
                self.observe(frame_index, 1, 150)

        self.assertEqual(output.getvalue().count("[异常滞留报警]"), 1)
        self.assertEqual(self.source.count_in, 0)
        self.assertEqual(self.source.count_out, 0)

    def test_missing_gate_fallback_does_not_match_recent_start_on_other_path(self):
        self.source.left_gate_y = (0, 500)
        self.source.right_gate_y = (0, 500)
        self.observe(1, 50, 90, center_y=90)
        self.observe(2, 50, 110, center_y=110)
        self.source.pending_directions.pop(50)

        self.observe(3, 54, 180, center_y=320)
        self.observe(4, 54, 195, center_y=330)
        self.observe(5, 54, 205, center_y=340)

        self.assertEqual(self.source.count_in, 1)

    def test_missing_gate_fallback_counts_recent_person_in_separate_lane(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150, center_y=80)
        self.observe(3, 1, 250, center_y=80)
        self.observe(30, 54, 190, center_y=240)
        self.observe(31, 54, 210, center_y=240)

        self.assertEqual(self.source.count_in, 2)

    def test_debug_reports_inferred_completion_without_first_gate_state(self):
        self.observe(1, 1, 150)
        self.source.frame_index = 2
        output = StringIO()
        with redirect_stdout(output):
            update_track_crossing(
                self.source,
                1,
                (70, 80, 90, 120),
                100,
                200,
                debug_counting=True,
            )

        self.assertIn("首门轨迹缺失", output.getvalue())
        self.assertEqual(self.source.count_out, 1)

    def test_reassociates_new_track_id_after_detection_gap(self):
        self.observe(1, 1, 250)
        self.observe(2, 1, 180)
        self.observe(142, 2, 80)

        self.assertEqual(self.source.count_out, 1)
        self.assertNotIn(1, self.source.pending_directions)

    def test_reassociates_track_id_switch_inside_channel(self):
        self.observe(1, 1, 250)
        self.observe(2, 1, 180)
        self.observe(3, 2, 150)
        self.observe(4, 2, 80)

        self.assertEqual(self.source.count_out, 1)

    def test_does_not_associate_pending_crossing_with_detection_far_past_exit_gate(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)

        self.source.frame_index = 3
        update_track_crossing(
            self.source,
            2,
            (890, 80, 910, 120),
            100,
            200,
        )

        self.assertEqual(self.source.count_in, 0)
        self.assertEqual(self.source.pending_directions, {1: "left-to-right"})

        self.observe(4, 3, 180)
        self.observe(5, 3, 210)

        self.assertEqual(self.source.count_in, 1)
        self.assertNotIn(1, self.source.pending_directions)

    def test_simultaneous_distant_person_does_not_consume_pending_track(self):
        self.source.left_gate_y = (0, 1080)
        self.source.right_gate_y = (0, 1080)
        self.source.frame_index = 6950
        self.source.pending_directions[55] = "left-to-right"
        self.source.last_center_x[55] = 668
        self.source.last_track_boxes[55] = (618, 432, 718, 586, 6950)
        self.source.last_center_x[57] = 1186.5
        self.source.last_track_boxes[57] = (1135, 442, 1238, 690, 6950)

        self.source.frame_index = 6951
        tracks = [
            (53, (616, 429, 737, 808)),
            (57, (1138, 438, 1238, 682)),
            (55, (626, 430, 729, 586)),
        ]
        update_nested_track_pairs(
            self.source, [track_id for track_id, _ in tracks],
            [box for _, box in tracks],
        )
        output = StringIO()
        with redirect_stdout(output):
            update_track_crossing(
                self.source, 57, tracks[1][1], 471, 923, debug_counting=True
            )

        self.assertEqual(self.source.count_in, 0)
        self.assertEqual(self.source.pending_directions[55], "left-to-right")
        self.assertIn("关联拒绝", output.getvalue())

    def test_rejects_large_jump_to_new_id_after_one_missing_frame(self):
        self.source.left_gate_y = (0, 1080)
        self.source.right_gate_y = (0, 1080)
        self.source.frame_index = 10
        self.source.pending_directions[55] = "left-to-right"
        self.source.last_center_x[55] = 668
        self.source.last_track_boxes[55] = (618, 432, 718, 586, 10)
        self.source.frame_index = 11
        update_track_crossing(
            self.source, 57, (1138, 438, 1238, 682), 471, 923
        )
        self.assertEqual(self.source.count_in, 0)
        self.assertEqual(self.source.pending_directions[55], "left-to-right")

    def test_separated_tracks_remain_distinct_after_pending_id_disappears(self):
        self.source.left_gate_y = (0, 1080)
        self.source.right_gate_y = (0, 1080)
        self.source.frame_index = 6950
        self.source.pending_directions[55] = "left-to-right"
        self.source.last_center_x[55] = 668
        self.source.last_track_boxes[55] = (618, 432, 718, 586, 6950)
        self.source.last_center_x[57] = 1186.5
        self.source.last_track_boxes[57] = (1135, 442, 1238, 690, 6950)
        for frame, tracks in [
            (6951, [(57, (1138, 438, 1238, 682)), (55, (626, 430, 729, 586))]),
            (6961, [(57, (1154, 436, 1256, 682)), (55, (676, 430, 795, 581))]),
            (6971, [(57, (1152, 410, 1268, 692))]),
        ]:
            self.source.frame_index = frame
            update_nested_track_pairs(
                self.source, [track_id for track_id, _ in tracks],
                [box for _, box in tracks],
            )
            for track_id, box in tracks:
                update_track_crossing(self.source, track_id, box, 471, 923)

        self.assertEqual(self.source.count_in, 0)
        self.assertEqual(self.source.pending_directions[55], "left-to-right")

    def test_simultaneous_separate_tracks_are_not_joined_after_approaching(self):
        self.source.frame_index = 1
        self.source.pending_directions[1] = "left-to-right"
        self.source.last_center_x[1] = 170
        self.source.last_track_boxes[1] = (160, 80, 180, 120, 1)
        self.source.last_center_x[2] = 230
        self.source.last_track_boxes[2] = (220, 80, 240, 120, 1)
        update_nested_track_pairs(
            self.source, [1, 2], [(160, 80, 180, 120), (220, 80, 240, 120)]
        )
        self.source.frame_index = 2
        update_nested_track_pairs(self.source, [2], [(205, 80, 225, 120)])
        self.observe(2, 2, 215)

        self.assertEqual(self.source.count_in, 0)
        self.assertEqual(self.source.pending_directions[1], "left-to-right")

    def test_separated_track_evidence_is_cleared_after_pending_finishes(self):
        self.source.pending_directions[1] = "left-to-right"
        self.source.frame_index = 1
        update_nested_track_pairs(
            self.source, [1, 2], [(100, 50, 150, 200), (250, 50, 300, 200)]
        )
        self.assertIn((1, 2), self.source.separated_track_pairs)
        self.source.pending_directions.pop(1)
        self.source.frame_index = 2
        update_nested_track_pairs(self.source, [], [])
        self.assertNotIn((1, 2), self.source.separated_track_pairs)

    def test_counts_only_one_person_when_duplicate_tracks_match_one_detection(self):
        self.observe(1, 1, 250)
        self.observe(1, 2, 250)
        self.observe(2, 1, 115)
        self.observe(2, 2, 115)
        self.observe(3, 3, 90)

        self.assertEqual(self.source.count_out, 1)
        self.assertNotIn(1, self.source.pending_directions)
        self.assertNotIn(2, self.source.pending_directions)

    def test_nested_upper_body_and_full_body_tracks_count_once(self):
        self.source.left_line = 498
        self.source.right_line = 828
        self.source.left_gate_y = (0, 1000)
        self.source.right_gate_y = (0, 1000)
        frames = [
            (1, (359, 438, 602, 830), (401, 449, 504, 587)),
            (2, (380, 440, 610, 840), (430, 450, 530, 588)),
            (3, (393, 449, 611, 851), (450, 452, 553, 588)),
        ]
        for frame, full_box, upper_box in frames:
            self.source.frame_index = frame
            update_nested_track_pairs(self.source, [44, 49], [full_box, upper_box])
            update_track_crossing(self.source, 44, full_box, 498, 828)
            update_track_crossing(self.source, 49, upper_box, 498, 828)

        self.source.frame_index = 4
        update_nested_track_pairs(self.source, [49], [(770, 434, 870, 570)])
        update_track_crossing(self.source, 49, (770, 434, 870, 570), 498, 828)
        self.source.frame_index = 5
        update_nested_track_pairs(self.source, [49], [(781, 434, 883, 570)])
        update_track_crossing(self.source, 49, (781, 434, 883, 570), 498, 828)
        self.source.frame_index = 11
        update_nested_track_pairs(self.source, [44], [(821, 437, 962, 783)])
        update_track_crossing(self.source, 44, (821, 437, 962, 783), 498, 828)

        self.assertEqual(self.source.count_in, 1)

    def test_nested_track_chain_clears_all_pending_after_id_switch(self):
        self.source.left_gate_y = (0, 1000)
        self.source.right_gate_y = (0, 1000)

        def frame(number, tracks):
            self.source.frame_index = number
            update_nested_track_pairs(
                self.source, [track_id for track_id, _ in tracks],
                [box for _, box in tracks],
            )
            for track_id, box in tracks:
                update_track_crossing(self.source, track_id, box, 512, 865)

        for number in range(1, 4):
            frame(number, [
                (15, (809, 446, 930, 711)),
                (19, (822, 447, 927, 565)),
            ])
        frame(4, [
            (15, (797, 444, 920, 720)),
            (19, (807, 446, 914, 567)),
        ])
        for number in range(5, 23):
            frame(number, [(19, (604, 465, 710, 586))])
        for number in range(23, 26):
            frame(number, [
                (19, (589, 463, 697, 586)),
                (21, (585, 460, 708, 759)),
            ])
        frame(26, [(21, (453, 467, 573, 780))])
        frame(27, [(21, (445, 467, 566, 781))])
        frame(39, [
            (21, (300, 460, 420, 800)),
            (27, (303, 474, 407, 598)),
        ])

        self.assertEqual(self.source.count_out, 1)
        self.assertNotIn(15, self.source.pending_directions)
        self.assertNotIn(19, self.source.pending_directions)

    def test_nested_pair_is_discarded_when_both_boxes_separate(self):
        for number in range(1, 4):
            self.source.frame_index = number
            update_nested_track_pairs(
                self.source, [1, 2], [(100, 50, 200, 300), (120, 55, 180, 150)]
            )
        self.assertIn((1, 2), self.source.nested_track_pairs)
        self.source.frame_index = 4
        update_nested_track_pairs(
            self.source, [1, 2], [(100, 50, 200, 300), (250, 55, 310, 150)]
        )
        self.assertNotIn((1, 2), self.source.nested_track_pairs)

    def test_nested_pair_expires_after_pending_timeout(self):
        for number in range(1, 4):
            self.source.frame_index = number
            update_nested_track_pairs(
                self.source, [1, 2], [(100, 50, 200, 300), (120, 55, 180, 150)]
            )
        self.source.pending_directions[1] = "left-to-right"
        self.source.frame_index = 504
        update_nested_track_pairs(self.source, [1], [(100, 50, 200, 300)])
        self.assertNotIn((1, 2), self.source.nested_track_pairs)

    def test_nested_pair_without_pending_expires_after_short_gap(self):
        for number in range(1, 4):
            self.source.frame_index = number
            update_nested_track_pairs(
                self.source, [1, 2], [(100, 50, 200, 300), (120, 55, 180, 150)]
            )
        self.source.frame_index = 14
        update_nested_track_pairs(self.source, [1], [(100, 50, 200, 300)])
        self.assertNotIn((1, 2), self.source.nested_track_pairs)

    def test_nested_pair_does_not_consume_opposite_direction_pending(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)
        self.observe(2, 2, 250)
        self.observe(3, 2, 180)
        self.source.nested_track_pairs[(1, 2)] = (3, 3)
        self.observe(4, 1, 210)

        self.assertEqual(self.source.count_in, 1)
        self.assertEqual(self.source.pending_directions.get(2), "right-to-left")

    def test_transient_nested_boxes_do_not_merge_separate_pending_tracks(self):
        self.observe(1, 1, 50)
        self.observe(1, 2, 50, center_y=140)
        self.observe(2, 1, 150)
        self.observe(2, 2, 150, center_y=140)
        self.source.frame_index = 3
        update_nested_track_pairs(
            self.source, [1, 2], [(190, 50, 230, 150), (195, 55, 215, 90)]
        )
        self.observe(3, 1, 210)
        self.assertIn(2, self.source.pending_directions)
        self.source.frame_index = 10
        update_nested_track_pairs(self.source, [2], [(200, 120, 220, 160)])
        self.observe(10, 2, 210, center_y=140)
        self.assertEqual(self.source.count_in, 2)

    def test_similar_sized_overlapping_tracks_are_not_merged(self):
        for frame, x in enumerate((50, 90, 150, 180), start=1):
            self.source.frame_index = frame
            boxes = [(x - 10, 80, x + 10, 120), (x - 9, 80, x + 11, 120)]
            update_nested_track_pairs(self.source, [1, 2], boxes)
            for track_id, box in zip((1, 2), boxes):
                update_track_crossing(self.source, track_id, box, 100, 200)

        self.assertNotIn((1, 2), self.source.nested_track_pairs)
        self.assertEqual(self.source.pending_directions[1], "left-to-right")
        self.assertEqual(self.source.pending_directions[2], "left-to-right")

    def test_suppresses_duplicate_track_started_at_same_gate_and_path(self):
        self.observe(1, 38, 250)
        self.observe(2, 38, 190)
        self.observe(3, 39, 250)
        self.observe(4, 39, 190)
        self.observe(5, 38, 140)
        self.observe(6, 38, 90)
        self.observe(13, 39, 40)

        self.assertEqual(self.source.count_out, 1)

    def test_one_person_crossing_does_not_consume_another_person_pending_state(self):
        self.source.left_gate_y = (0, 319)
        self.source.right_gate_y = (0, 319)
        self.observe(1, 1, 50, center_y=100)
        self.observe(1, 2, 55, center_y=140)
        self.observe(2, 1, 150, center_y=100)
        self.observe(2, 2, 195, center_y=140)

        self.observe(3, 1, 220, center_y=100)
        self.assertIn(2, self.source.pending_directions)

        self.observe(10, 2, 220, center_y=140)
        self.assertEqual(self.source.count_in, 2)

    def test_suppresses_immediate_duplicate_crossing_after_track_id_switch(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)
        self.observe(3, 1, 250)
        self.observe(4, 7, 50)
        self.observe(5, 7, 150)
        self.observe(6, 7, 250)

        self.assertEqual(self.source.count_in, 1)

    def test_duplicate_crossing_does_not_consume_another_person_pending_state(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)
        self.observe(3, 1, 250)

        self.observe(4, 2, 50, center_y=140)
        self.observe(5, 2, 150, center_y=140)

        self.observe(6, 7, 150)
        self.observe(7, 7, 250)
        self.assertIn(2, self.source.pending_directions)
        self.assertEqual(self.source.count_in, 1)

        self.observe(10, 2, 250, center_y=140)
        self.assertEqual(self.source.count_in, 2)

    def assert_opposite_moving_track_does_not_consume_pending_entry(self, wide_box):
        self.observe(1, 55, 50)
        self.observe(2, 55, 120)
        self.observe(3, 55, 170)
        self.observe(4, 57, 340)

        if wide_box:
            self.observe_box(5, 57, (180, 80, 260, 120))
        else:
            self.observe(5, 57, 220)

        self.assertEqual(self.source.count_in, 0)
        self.assertEqual(self.source.pending_directions[55], "left-to-right")

        self.observe(6, 57, 190)
        self.observe(7, 57, 90)
        self.observe(8, 55, 220)
        self.assertEqual(self.source.count_in, 1)
        self.assertEqual(self.source.count_out, 1)

    def test_opposite_motion_blocks_pending_track_box_association(self):
        self.assert_opposite_moving_track_does_not_consume_pending_entry(
            wide_box=True
        )

    def test_opposite_motion_blocks_pending_track_gate_fallback(self):
        self.assert_opposite_moving_track_does_not_consume_pending_entry(
            wide_box=False
        )

    def test_recent_motion_blocks_false_reverse_association_after_small_last_step(self):
        self.source.left_gate_y = (0, 1080)
        self.source.right_gate_y = (0, 1080)
        boxes = [
            (9881, (632, 433, 821, 784)),
            (9882, (634, 432, 825, 788)),
            (9883, (639, 429, 830, 789)),
            (9884, (641, 427, 830, 788)),
            (9885, (635, 424, 820, 786)),
            (9886, (649, 420, 835, 788)),
            (9887, (654, 416, 839, 785)),
            (9888, (659, 411, 842, 777)),
            (9889, (663, 408, 847, 774)),
            (9890, (667, 405, 851, 771)),
            (9892, (669, 401, 860, 784)),
        ]
        for frame, box in boxes:
            self.source.frame_index = frame
            update_track_crossing(self.source, 392, box, 868, 1014)

        self.source.pending_directions[349] = "right-to-left"
        self.source.last_center_x[349] = 989.5
        self.source.last_track_boxes[349] = (952, 159, 1027, 289, 9777)
        self.source.frame_index = 9893
        output = StringIO()
        with redirect_stdout(output):
            update_track_crossing(
                self.source, 392, (670, 399, 863, 786), 868, 1014,
                debug_counting=True,
            )

        self.assertEqual(self.source.count_out, 0)
        self.assertEqual(self.source.pending_directions[349], "right-to-left")
        self.assertIn("运动方向与待计方向相反", output.getvalue())

    def test_recent_motion_blocks_opposite_box_association_in_both_directions(self):
        for direction in ("left-to-right", "right-to-left"):
            with self.subTest(direction=direction):
                source = SourceState(
                    name="test", window_name="test", capture=cv2.VideoCapture(),
                    left_gate_y=(0, 1080), right_gate_y=(0, 1080),
                )
                try:
                    positions = [250, 240, 230, 220]
                    if direction == "right-to-left":
                        positions = [300 - x for x in positions]
                    for frame, x in enumerate(positions, 1):
                        source.frame_index = frame
                        update_track_crossing(
                            source, 2, (x - 10, 80, x + 10, 120), 100, 200
                        )
                    source.pending_directions[1] = direction
                    source.last_center_x[1] = 180 if direction == "left-to-right" else 120
                    source.last_track_boxes[1] = (
                        source.last_center_x[1] - 10, 80,
                        source.last_center_x[1] + 10, 120, 4,
                    )
                    source.frame_index = 5
                    x = 218 if direction == "left-to-right" else 82
                    update_track_crossing(
                        source, 2, (x - 60, 80, x + 60, 120), 100, 200
                    )
                    self.assertEqual(source.count_in + source.count_out, 0)
                    self.assertEqual(source.pending_directions[1], direction)
                finally:
                    source.capture.release()

    def test_recent_motion_preserves_matching_id_switch_after_small_last_step(self):
        for frame, x in enumerate((90, 80, 70, 60), 1):
            self.observe(frame, 2, x)
        self.source.pending_directions[1] = "right-to-left"
        self.source.last_center_x[1] = 150
        self.source.last_track_boxes[1] = (140, 80, 160, 120, 4)

        self.observe(5, 2, 58)

        self.assertEqual(self.source.count_out, 1)
        self.assertNotIn(1, self.source.pending_directions)

    def test_old_motion_does_not_block_id_switch_after_detection_gap(self):
        for frame, x in enumerate((250, 240, 230), 1):
            self.observe(frame, 2, x)
        self.source.pending_directions[1] = "left-to-right"
        self.source.last_center_x[1] = 180
        self.source.last_track_boxes[1] = (170, 80, 190, 120, 29)

        self.observe(30, 2, 231)

        self.assertEqual(self.source.count_in, 1)
        self.assertNotIn(1, self.source.pending_directions)

    def test_new_id_does_not_reuse_reverse_pending_after_opposite_track_disappears(self):
        self.source.left_gate_y = (0, 1080)
        self.source.right_gate_y = (0, 1080)
        for frame, box in [
            (9887, (654, 416, 839, 785)),
            (9888, (659, 411, 842, 777)),
            (9889, (663, 408, 847, 774)),
            (9890, (667, 405, 851, 771)),
            (9892, (669, 401, 860, 784)),
            (9893, (670, 399, 863, 786)),
            (9894, (677, 399, 863, 775)),
            (9895, (678, 398, 865, 780)),
            (9896, (701, 397, 851, 695)),
            (9897, (710, 396, 848, 663)),
            (9898, (704, 313, 867, 631)),
            (9899, (730, 292, 873, 575)),
        ]:
            self.source.frame_index = frame
            update_track_crossing(self.source, 392, box, 868, 1014)
        self.source.pending_directions[349] = "right-to-left"
        self.source.last_center_x[349] = 989.5
        self.source.last_track_boxes[349] = (952, 159, 1027, 289, 9777)

        self.source.frame_index = 9904
        update_track_crossing(
            self.source, 400, (709, 372, 879, 739), 868, 1014
        )
        self.assertEqual(self.source.count_out, 0)
        self.assertEqual(self.source.pending_directions[349], "right-to-left")

        for frame, box in [
            (9905, (718, 369, 878, 716)),
            (9906, (723, 367, 880, 709)),
            (9907, (723, 366, 887, 724)),
            (9925, (785, 350, 955, 700)),
            (9981, (935, 200, 1105, 500)),
        ]:
            self.source.frame_index = frame
            update_track_crossing(self.source, 400, box, 868, 1014)
        self.assertEqual(self.source.count_out, 0)
        self.assertEqual(self.source.count_in, 1)

    def test_visible_opposite_track_does_not_block_another_new_track(self):
        self.source.left_gate_y = (0, 319)
        self.source.right_gate_y = (0, 319)
        for frame, x in enumerate((50, 60, 70), 1):
            self.observe(frame, 2, x)
        self.source.pending_directions[1] = "right-to-left"
        self.source.last_center_x[1] = 150
        self.source.last_track_boxes[1] = (140, 80, 160, 120, 3)
        self.source.frame_index = 4
        boxes = [(65, 80, 85, 120), (64, 80, 84, 120)]
        update_nested_track_pairs(self.source, [2, 3], boxes)

        update_track_crossing(self.source, 3, boxes[1], 100, 200)

        self.assertEqual(self.source.count_out, 1)

    def test_new_track_with_confirmed_matching_motion_can_complete(self):
        for frame, x in enumerate((50, 60, 70), 1):
            self.observe(frame, 2, x)
        self.source.pending_directions[1] = "right-to-left"
        self.source.last_center_x[1] = 150
        self.source.last_track_boxes[1] = (140, 80, 160, 120, 3)

        self.observe(4, 3, 75)
        self.assertEqual(self.source.count_out, 0)
        self.observe(5, 3, 60)

        self.assertEqual(self.source.count_out, 1)
        self.assertNotIn(1, self.source.pending_directions)

    def test_previously_counted_track_cannot_reuse_another_tracks_pending_state(self):
        self.observe(1, 1, 250)
        self.observe(2, 1, 150)
        self.observe(3, 1, 50)
        self.assertEqual(self.source.count_out, 1)

        self.observe(4, 2, 250)
        self.observe(5, 2, 150)
        self.assertEqual(self.source.pending_directions[2], "right-to-left")

        self.observe_box(10, 1, (-50, 80, 150, 120))
        self.assertEqual(self.source.count_out, 1)
        self.assertEqual(self.source.pending_directions[2], "right-to-left")

    def test_same_track_can_count_again_after_a_new_first_gate_crossing(self):
        self.observe(1, 1, 250)
        self.observe(2, 1, 150)
        self.observe(3, 1, 50)
        self.observe(4, 1, 150)
        self.observe(5, 1, 250)
        self.observe(10, 1, 150)
        self.observe(11, 1, 50)

        self.assertEqual(self.source.count_out, 2)

    def test_counts_separate_people_following_the_same_path(self):
        self.source.left_gate_y = (90, 150)
        self.source.right_gate_y = (90, 150)
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)
        self.observe(3, 1, 250)
        self.observe(30, 7, 50)
        self.observe(31, 7, 150)
        self.observe(32, 7, 250)

        self.assertEqual(self.source.count_in, 2)

    def test_does_not_count_reversal_inside_gate(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)
        self.observe(3, 1, 90)

        self.assertEqual(self.source.count_in, 0)
        self.assertEqual(self.source.count_out, 0)

    def test_does_not_count_crossings_outside_finite_gate_segments(self):
        self.source.left_gate_y = (250, 300)
        self.source.right_gate_y = (250, 300)
        self.observe(1, 1, 50, center_y=200)
        self.observe(2, 1, 150, center_y=200)
        self.observe(3, 1, 250, center_y=200)

        self.assertEqual(self.source.count_in, 0)

    def test_counts_when_both_gate_segments_are_crossed(self):
        self.source.left_gate_y = (90, 150)
        self.source.right_gate_y = (90, 150)
        self.observe(1, 1, 50, center_y=90)
        self.observe(2, 1, 150, center_y=90)
        self.observe(3, 1, 250, center_y=90)

        self.assertEqual(self.source.count_in, 1)

    def test_cancels_pending_crossing_when_second_gate_is_crossed_outside_segment(self):
        self.source.left_gate_y = (100, 160)
        self.source.right_gate_y = (100, 160)
        self.observe(1, 1, 50, center_y=90)
        self.observe(2, 1, 150, center_y=90)
        self.observe(3, 1, 250, center_y=220)

        self.assertEqual(self.source.count_in, 0)
        self.assertNotIn(1, self.source.pending_directions)

    def test_does_not_count_when_first_gate_is_crossed_outside_segment(self):
        self.source.left_gate_y = (250, 300)
        self.source.right_gate_y = (90, 150)
        self.observe(1, 1, 50, center_y=200)
        self.observe(2, 1, 150, center_y=200)
        self.observe(3, 1, 250, center_y=200)

        self.assertEqual(self.source.count_in, 0)
        self.assertNotIn(1, self.source.pending_directions)

    def test_does_not_guess_between_ambiguous_pending_tracks(self):
        self.observe(1, 1, 250)
        self.observe(1, 2, 260)
        self.observe(2, 1, 180)
        self.observe(2, 2, 190)
        self.observe(3, 3, 80)

        self.assertEqual(self.source.count_out, 0)


if __name__ == "__main__":
    unittest.main()
