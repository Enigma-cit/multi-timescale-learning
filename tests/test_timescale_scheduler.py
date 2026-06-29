from mtsl_engine.core.timescale_scheduler import TimescaleScheduler


def test_scheduler_updates_flags():
    sched = TimescaleScheduler(short_every=1, medium_every=4, long_every=16)
    short_flags = []
    medium_flags = []
    long_flags = []

    for _ in range(32):
        short_flags.append(sched.should_update_short())
        medium_flags.append(sched.should_update_medium())
        long_flags.append(sched.should_update_long())
        sched.advance()

    # Short should be True at every step
    assert all(short_flags)

    # Medium should be True at multiples of 4 (0,4,8,...)
    for i, flag in enumerate(medium_flags):
        if i % 4 == 0:
            assert flag is True
        else:
            assert flag is False

    # Long should be True at multiples of 16 (0,16,32,...)
    for i, flag in enumerate(long_flags):
        if i % 16 == 0:
            assert flag is True
        else:
            assert flag is False