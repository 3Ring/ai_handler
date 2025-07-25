from ai_handler.cache import InMemoryCache, NullCache
from ai_handler.question import SimpleQuestion

def test_in_memory_cache_set_and_get():
    cache = InMemoryCache()
    q = SimpleQuestion("foo?")
    cache.set(q, "bar!")
    assert cache.get(q) == "bar!"

def test_in_memory_cache_returns_none_for_missing():
    cache = InMemoryCache()
    q = SimpleQuestion("missing?")
    assert cache.get(q) is None

def test_null_cache_always_misses():
    cache = NullCache()
    q = SimpleQuestion("irrelevant?")
    cache.set(q, "should not store")
    assert cache.get(q) is None
