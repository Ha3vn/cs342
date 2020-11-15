import sys


class CheckFailed(Exception):
    def __init__(self, why):
        self.why = why

    def __str__(self):
        return self.why


class ContextManager:
    def __init__(self, on, off):
        self.on = on
        self.off = off

    def __enter__(self):
        self.on()

    def __exit__(self, exc_type, exc_value, traceback):
        self.off()


def list_all_kwargs(**kwargs):
    all_args = [{}]
    for k, v in kwargs.items():
        new_args = []
        for i in v:
            new_args.extend([dict({k: i}, **a) for a in all_args])
        all_args = new_args
    return all_args


# Use @Case(score, extra_credit) as a decorator for member functions of a Grader
# this will make them test cases
# A test case can return a value between 0 and 1 as a score. If the test fails it should raise an assertion.
# The test case may optionally return a tuple (score, message).


def case(func, kwargs={}, score=1, extra_credit=False):
    def wrapper(self):
        msg = 'passed'
        n_passed, total = 0.0, 0.0
        for a in list_all_kwargs(**kwargs):
            try:
                v = func(self, **a)
                if v is None:
                    v = 1
                elif isinstance(v, tuple):
                    v, msg = v
                else:
                    assert isinstance(v, float), "case returned %s which is not a float!" % repr(v)
                n_passed += v
            except AssertionError as e:
                msg = str(e)
            except CheckFailed as e:
                msg = str(e)
            except NotImplementedError as e:
                msg = 'Function not implemented %s' % e
            except Exception as e:
                msg = 'Crash "%s"' % e
            total += 1
        return int(n_passed * score / total + 0.5), msg

    wrapper.score = score
    wrapper.extra_credit = extra_credit
    wrapper.__doc__ = func.__doc__
    return wrapper


class Case(object):
    def __init__(self, score=1, extra_credit=False):
        self.score = score
        self.extra_credit = extra_credit

    def __call__(self, func):
        return case(func, score=self.score, extra_credit=self.extra_credit)


class MultiCase(object):
    def __init__(self, score=1, extra_credit=False, **kwargs):
        self.score = score
        self.extra_credit = extra_credit
        self.kwargs = kwargs

    def __call__(self, func):
        return case(func, kwargs=self.kwargs, score=self.score, extra_credit=self.extra_credit)


class Grader:
    def __init__(self, module, verbose=False):
        self.module = module
        self.verbose = verbose

    @classmethod
    def has_cases(cls):
        import inspect
        for n, f in inspect.getmembers(cls):
            if hasattr(f, 'score'):
                return True
        return False

    @classmethod
    def total_score(cls):
        import inspect
        r = 0
        for n, f in inspect.getmembers(cls):
            if hasattr(f, 'score'):
                r += f.score
        return r

    def run(self, f_out):
        import inspect
        score, total_score = 0, 0
        if self.verbose:
            print(' * %-50s' % self.__doc__, file=f_out)
        for n, f in inspect.getmembers(self):
            if hasattr(f, 'score'):
                s, msg = f()
                score += s
                if self.verbose:
                    print('  - %-50s [ %s ]' % (f.__doc__, msg), file=f_out)
                if not f.extra_credit:
                    total_score += f.score

        return score, total_score


def grade(G, assignment_module, verbose, f_out):
    try:
        grader = G(assignment_module, verbose)
    except NotImplementedError as e:
        if verbose:
            print('  - Function not implemented: %s' % e, file=f_out)
        return 0, G.total_score()
    except Exception as e:
        if verbose:
            print('  - Your program crashed "%s"' % e, file=f_out)
        return 0, G.total_score()

    return grader.run(f_out)


def grade_all(assignment_module, verbose=False, f_out=sys.stdout):
    score, total_score = 0, 0
    for G in Grader.__subclasses__():
        if G.has_cases():
            s, ts = grade(G, assignment_module, verbose, f_out=f_out)

            if verbose:
                print(' --------------------------------------------------    [ %3d / %3d ]' % (s, ts), file=f_out)
                print(file=f_out)
            else:
                print(' * %-50s  [ %3d / %3d ]' % (G.__doc__, s, ts), file=f_out)
            total_score += ts
            score += s

    print(file=f_out)
    print('total score                                              %3d / %3d' % (score, total_score), file=f_out)
    return score


def load_assignment(name, f_out=sys.stdout, pre_import_fn=None):
    import atexit
    import importlib
    from pathlib import Path
    from shutil import rmtree
    import sys
    import tempfile
    import zipfile

    if Path(name).is_dir():
        if pre_import_fn is not None:
            pre_import_fn()
        return importlib.import_module(name)

    with zipfile.ZipFile(name) as f:
        tmp_dir = Path(tempfile.mkdtemp())
        atexit.register(lambda: rmtree(tmp_dir))

        f.extractall(tmp_dir)
        module_names = list(tmp_dir.glob('*/'))
        if len(module_names) != 1:
            print('Malformed zip file, expecting exactly one top-level folder, got %d' % len(module_names), file=f_out)
            return None
        sys.path.insert(0, str(tmp_dir))
        module = module_names[0].name
        if pre_import_fn is not None:
            pre_import_fn()
        try:
            return importlib.import_module(module)
        except ModuleNotFoundError as e:
           print('Import error "%s"'%str(e), file=f_out)
        except e:
           print('Failed to load your solution: "%s"'%str(e), file=f_out)


def run():
    import argparse

    parser = argparse.ArgumentParser('Grade your assignment')
    parser.add_argument('assignment', default='homework')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    print('Loading assignment')
    assignment = load_assignment(args.assignment)
    if assignment is None:
        return 0

    print('Loading grader')
    total_score = grade_all(assignment, args.verbose)
    return total_score
