from . import tests

try:
    # For test grading only, don't worry about it being missing ;)
    from .safe_grader import run
except ImportError:
    from .grader import run

run()
