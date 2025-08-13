import os
import signal
import dotenv
from pathlib import Path
from typing import Optional


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

# This is a decorator to add timeout functionality
def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and a 5-second alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)

            try:
                # This try/except clause is used to catch the TimeoutError
                # and to turn off the alarm in case the function finished
                # before the timeout
                result = func(*args, **kwargs)
            except TimeoutError as te:
                print(te, seconds)
                return None
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result

        return wrapper

    return decorator

def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(
                f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value

def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)
