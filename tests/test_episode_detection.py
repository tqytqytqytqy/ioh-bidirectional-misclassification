import numpy as np

from ioh.estimands.episodes import detect_reference_episodes, episode_documented


def test_short_episode_between_sampling_points_is_not_mechanically_detected():
    times = np.arange(0, 130, 10, dtype=float)
    reference = np.full_like(times, 75.0)
    reference[(times >= 30) & (times <= 50)] = 60.0
    display = np.full_like(times, 75.0)

    episodes = detect_reference_episodes(times, reference, threshold=65.0, min_duration_sec=0, gap_tolerance_sec=0)

    assert len(episodes) == 1
    assert episode_documented(episodes[0], times, display, threshold=65.0, grace_after_sec=0) is False


def test_gap_tolerance_merges_brief_recovery_inside_reference_episode():
    times = np.arange(0, 100, 10, dtype=float)
    reference = np.array([75, 60, 60, 70, 60, 60, 75, 75, 75, 75], dtype=float)

    no_merge = detect_reference_episodes(times, reference, threshold=65.0, min_duration_sec=0, gap_tolerance_sec=0)
    merged = detect_reference_episodes(times, reference, threshold=65.0, min_duration_sec=0, gap_tolerance_sec=20)

    assert len(no_merge) == 2
    assert len(merged) == 1
