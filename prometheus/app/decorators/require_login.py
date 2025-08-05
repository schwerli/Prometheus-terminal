from functools import wraps


def requireLogin(func):
    """
    Decorator to indicate that a route requires user authentication.
    This decorator can be used to mark routes that should only be accessible to authenticated users.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    # Set a custom attribute to indicate that this route requires login
    setattr(wrapper, "_require_login", True)
    return wrapper
