from rate_limit import RateLimiter


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_per_client_limit_slides_with_time():
    clock = Clock()
    limiter = RateLimiter(per_minute=2, per_day=100, clock=clock)
    assert limiter.check("a") == (True, 0)
    clock.now += 10
    assert limiter.check("a") == (True, 0)
    allowed, wait = limiter.check("a")
    assert not allowed and wait == 51  # the first request leaves the window 50 s from now
    assert limiter.check("b") == (True, 0)  # other clients are not affected
    clock.now += 51
    assert limiter.check("a") == (True, 0)


def test_global_daily_cap_applies_to_everyone():
    clock = Clock()
    limiter = RateLimiter(per_minute=100, per_day=3, clock=clock)
    assert all(limiter.check(f"client-{i}")[0] for i in range(3))
    allowed, wait = limiter.check("someone-new")  # a fresh IP doesn't get around it
    assert not allowed and wait > 86_000
    clock.now += 86_400
    assert limiter.check("someone-new") == (True, 0)


def test_refused_requests_are_not_counted():
    clock = Clock()
    limiter = RateLimiter(per_minute=1, per_day=100, clock=clock)
    limiter.check("a")
    for _ in range(5):
        limiter.check("a")  # refused
    clock.now += 60
    assert limiter.check("a") == (True, 0)
