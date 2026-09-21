class GenerationCancelled(Exception):
    """用户主动取消了正在进行的生成任务。"""
    pass


class GeneratorBusyError(Exception):
    """已有生成任务占用着生成器：单驻留下此时不能切换、卸载模型，也不能并发使用同一个实例。"""
    pass
