import tests.sample_objects as so
import ajmc.commons.arithmetic as ar


def test_compute_inter_overlap():
    assert ar.compute_interval_overlap(so.sample_intervals['base'], so.sample_intervals['included']) == 7
    assert ar.compute_interval_overlap(so.sample_intervals['base'], so.sample_intervals['overlapping']) == 3
    assert ar.compute_interval_overlap(so.sample_intervals['base'], so.sample_intervals['non_overlapping']) == 0


def test_is_interval_within_interval():
    assert ar.is_interval_within_interval(so.sample_intervals['included'], so.sample_intervals['base'])
    assert ar.is_interval_within_interval(so.sample_intervals['base'], so.sample_intervals['base'])
    assert not ar.is_interval_within_interval(so.sample_intervals['overlapping'], so.sample_intervals['base'])
    assert not ar.is_interval_within_interval(so.sample_intervals['non_overlapping'], so.sample_intervals['base'])


def test_are_intervals_within_intervals():
    assert ar.are_intervals_within_intervals(so.sample_interval_lists['included'], so.sample_interval_lists['base'])
    assert ar.are_intervals_within_intervals(so.sample_interval_lists['base'], so.sample_interval_lists['base'])
    assert not ar.are_intervals_within_intervals(so.sample_interval_lists['overlapping'],
                                                 so.sample_interval_lists['base'])
    assert not ar.are_intervals_within_intervals(so.sample_interval_lists['non_overlapping'],
                                                 so.sample_interval_lists['base'])

