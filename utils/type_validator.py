"""
The type decorator raise an exception if the provided type of an attribute of
the decorated function is different from the annoted type in the function
signature
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# to get information from the decorated function
import inspect
# to keep decorated function doc string
from functools import wraps


# -----------------------------------------------------------------------------
# type validator
# -----------------------------------------------------------------------------
# generic type validation based on type annotation in function signature
def type_validator(func):
    """
    Decorator that will rely on the types and attributes declaration in the
    function signature to check the actual types of the parameter against the
    expected types
    """
    # extract information about the function's parameters and return type.
    sig = inspect.signature(func)
    # preserve name and docstring of decorated function
    @wraps(func)
    def wrapper(*args, **kwargs):
        # map the parameter from signature to their corresponding values
        bound_args = sig.bind(*args, **kwargs)
        # check for each name of param if value has the declared type
        for name, value in bound_args.arguments.items():
            if name in sig.parameters:
                param = sig.parameters[name]
                if (param.annotation != param.empty
                        and not isinstance(value, param.annotation)):
                    raise TypeError(f"function '{func.__name__}' : expected "
                                    f"type '{param.annotation}' for argument "
                                    f"'{name}' but got {type(value)}.")
        return func(*args, **kwargs)
    return wrapper
