from functools import wraps


def requireLogin(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    # Set a custom attribute to indicate that this route requires login
    setattr(wrapper, "_require_login", True)
    return wrapper
