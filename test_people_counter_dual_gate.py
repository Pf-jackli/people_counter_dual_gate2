import cv2
import unittest

from people_counter_dual_gate import (
    SourceState,
    build_roi_points,
    total_counts,
    update_track_crossing,
    validate_roi_positions,
)


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
            roi_shape="rectangle",
            roi_points=[(50, 20), (250, 20), (250, 300), (50, 300)],
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

    def tearDown(self):
        self.source.capture.release()

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

    def test_does_not_count_track_first_detected_inside_gate_when_it_exits(self):
        self.observe(1, 1, 150)
        self.observe(2, 1, 80)

        self.assertEqual(self.source.count_out, 0)

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

    def test_counts_only_one_person_when_duplicate_tracks_match_one_detection(self):
        self.observe(1, 1, 250)
        self.observe(1, 2, 250)
        self.observe(2, 1, 115)
        self.observe(2, 2, 115)
        self.observe(3, 3, 90)

        self.assertEqual(self.source.count_out, 1)
        self.assertNotIn(1, self.source.pending_directions)
        self.assertNotIn(2, self.source.pending_directions)

    def test_suppresses_duplicate_crossing_after_track_id_switch(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)
        self.observe(3, 1, 250)
        self.observe(30, 7, 50)
        self.observe(31, 7, 150)
        self.observe(32, 7, 250)

        self.assertEqual(self.source.count_in, 1)

    def test_counts_a_separate_person_at_a_different_position(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)
        self.observe(3, 1, 250)
        self.observe(30, 7, 50, center_y=220)
        self.observe(31, 7, 150, center_y=220)
        self.observe(32, 7, 250, center_y=220)

        self.assertEqual(self.source.count_in, 2)

    def test_does_not_count_reversal_inside_gate(self):
        self.observe(1, 1, 50)
        self.observe(2, 1, 150)
        self.observe(3, 1, 90)

        self.assertEqual(self.source.count_in, 0)
        self.assertEqual(self.source.count_out, 0)

    def test_does_not_count_crossings_outside_channel_roi(self):
        self.source.roi_points = [
            (80, 60),
            (220, 60),
            (220, 160),
            (80, 160),
        ]
        self.observe(1, 1, 50, center_y=200)
        self.observe(2, 1, 150, center_y=200)
        self.observe(3, 1, 250, center_y=200)

        self.assertEqual(self.source.count_in, 0)

    def test_counts_when_detection_foot_is_just_outside_roi_edge(self):
        self.source.roi_points = [
            (50, 20),
            (250, 20),
            (250, 100),
            (50, 100),
        ]
        self.source.frame_index = 1
        update_track_crossing(
            self.source, 1, (40, 50, 60, 90), 100, 200
        )
        self.source.frame_index = 2
        update_track_crossing(
            self.source, 1, (140, 50, 160, 100), 100, 200
        )
        self.source.frame_index = 3
        update_track_crossing(
            self.source, 1, (240, 50, 260, 104), 100, 200
        )

        self.assertEqual(self.source.count_in, 1)

    def test_requires_both_gate_crossings_to_be_inside_channel_roi(self):
        self.source.roi_points = [
            (80, 80),
            (220, 80),
            (220, 130),
            (80, 180),
        ]
        self.observe(1, 1, 50, center_y=100)
        self.observe(2, 1, 150, center_y=150)
        self.observe(3, 1, 250, center_y=170)

        self.assertEqual(self.source.count_in, 0)
        self.assertNotIn(1, self.source.pending_directions)

    def test_parallelogram_roi_accepts_complete_crossing_inside_corridor(self):
        self.source.roi_shape = "parallelogram"
        self.source.roi_points = build_roi_points(
            "parallelogram", [(80, 60), (220, 100), (220, 220)]
        )
        self.observe(1, 1, 50, center_y=80)
        self.observe(2, 1, 150, center_y=120)
        self.observe(3, 1, 250, center_y=160)

        self.assertEqual(self.source.count_in, 1)

    def test_rejects_roi_that_does_not_cover_both_counting_lines(self):
        self.source.roi_points = [(20, 20), (90, 20), (90, 200), (20, 200)]

        error = validate_roi_positions(self.source, self.source.roi_points)

        self.assertIsNotNone(error)

    def test_does_not_guess_between_ambiguous_pending_tracks(self):
        self.observe(1, 1, 250)
        self.observe(1, 2, 260)
        self.observe(2, 1, 180)
        self.observe(2, 2, 190)
        self.observe(3, 3, 80)

        self.assertEqual(self.source.count_out, 0)


if __name__ == "__main__":
    unittest.main()
