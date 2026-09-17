import threading
from typing import Callable, Any

class BackgroundPreloader:
    """
    启动后立即在后台线程加载重型模块。
    调用 get() 时：已就绪则直接返回，未就绪则等待剩余时间。
    """
    def __init__(self):
        self._store: dict[str, Any] = {}
        self._events: dict[str, threading.Event] = {}
        self._errors: dict[str, Exception] = {}
        self._start_lock = threading.Lock()

    def preload(self, name: str, loader: Callable[[], Any]) -> None:
        """注册并立即在后台线程开始加载"""
        event = threading.Event()
        self._events[name] = event

        def _run():
            try:
                print(f"[Preloader] 开始加载: {name}")
                self._store[name] = loader()
                print(f"[Preloader] 加载完成: {name}")
            except Exception as e:
                self._errors[name] = e
                print(f"[Preloader] 加载失败 {name}: {e}")
            finally:
                event.set()
        threading.Thread(target=_run, daemon=True, name=f"preload-{name}").start()

    def get(self, name: str) -> Any:
        """获取模块，未就绪时阻塞等待"""
        if name not in self._events:
            raise KeyError(f"'{name}' 未注册到 Preloader")

        self._events[name].wait()
        if name in self._errors:
            raise self._errors[name]

        return self._store[name]

    def is_ready(self, name: str) -> bool:
        event = self._events.get(name)
        return event is not None and event.is_set()

    def load(self, name: str, loader: Callable[[], Any]) -> Any:
        """
        懒加载：首次调用时才在后台线程启动加载（若尚未启动），随后阻塞等待并返回结果。
        之后的调用直接复用已启动/已完成的加载，不会重复触发。
        """
        if name not in self._events:
            with self._start_lock:
                if name not in self._events:
                    self.preload(name, loader)
        return self.get(name)



preloader = BackgroundPreloader()