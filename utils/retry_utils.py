"""
重试工具模块 - 提供带指数退避的异步重试机制

用于处理网络波动、API超时等临时性错误
"""

import asyncio
import random
import logging
from typing import Tuple, Type, Callable, Any, Optional

logger = logging.getLogger(__name__)


async def with_retry(
    func: Callable,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Any:
    """
    带指数退避的异步重试机制

    Args:
        func: 要执行的异步函数（无参数，使用 lambda 包装带参函数）
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数基数
        jitter: 是否添加随机抖动（避免惊群效应）
        retryable_exceptions: 可重试的异常类型元组
        on_retry: 重试时的回调函数 (exception, attempt_number) -> None

    Returns:
        函数执行结果

    Raises:
        最后一次重试的异常

    Example:
        result = await with_retry(
            lambda: api_call(param1, param2),
            max_retries=3,
            retryable_exceptions=(ConnectionError, TimeoutError)
        )
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                # 计算延迟（带抖动）
                actual_delay = delay
                if jitter:
                    actual_delay += random.uniform(0, delay * 0.1)
                actual_delay = min(actual_delay, max_delay)

                logger.warning(
                    f"重试 {attempt + 1}/{max_retries}: {type(e).__name__}: {e}. "
                    f"等待 {actual_delay:.1f}s 后重试..."
                )

                if on_retry:
                    try:
                        on_retry(e, attempt + 1)
                    except Exception:
                        pass  # 忽略回调错误

                await asyncio.sleep(actual_delay)
                delay *= exponential_base
            else:
                logger.error(f"已达最大重试次数 ({max_retries}), 放弃: {e}")
                raise last_exception

    # 理论上不会到达这里
    if last_exception:
        raise last_exception


def retry_decorator(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    重试装饰器

    Example:
        @retry_decorator(max_retries=3, retryable_exceptions=(ConnectionError,))
        async def my_api_call():
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await with_retry(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
            )
        return wrapper
    return decorator


# 可重试的网络相关异常
NETWORK_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    OSError,  # 包含各种网络错误
)

# 尝试导入 httpx 异常
try:
    import httpx
    NETWORK_EXCEPTIONS = NETWORK_EXCEPTIONS + (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.NetworkError,
    )
except ImportError:
    pass

# 尝试导入 anthropic 异常
try:
    from anthropic import APIConnectionError, APITimeoutError
    NETWORK_EXCEPTIONS = NETWORK_EXCEPTIONS + (
        APIConnectionError,
        APITimeoutError,
    )
except ImportError:
    pass

# 尝试导入 openai 异常
try:
    from openai import APIConnectionError as OpenAIConnectionError
    from openai import APITimeoutError as OpenAITimeoutError
    NETWORK_EXCEPTIONS = NETWORK_EXCEPTIONS + (
        OpenAIConnectionError,
        OpenAITimeoutError,
    )
except ImportError:
    pass


def is_rate_limit_error(exception: Exception) -> bool:
    """检查是否为速率限制错误"""
    # 检查状态码
    if hasattr(exception, 'status_code'):
        if exception.status_code in (429, 503):
            return True

    # 检查响应对象
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        if exception.response.status_code in (429, 503):
            return True

    # 检查错误消息
    error_msg = str(exception).lower()
    if 'rate limit' in error_msg or 'too many requests' in error_msg:
        return True

    return False


def get_retry_after(exception: Exception, default: float = 10.0) -> float:
    """从异常中获取 retry-after 时间"""
    # 尝试从异常属性获取
    if hasattr(exception, 'retry_after'):
        return float(exception.retry_after)

    # 尝试从响应头获取
    if hasattr(exception, 'response') and hasattr(exception.response, 'headers'):
        retry_after = exception.response.headers.get('retry-after')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    return default
