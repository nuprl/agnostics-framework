set venv_dir ".venv"
if test -d $venv_dir
    source "$venv_dir/bin/activate.fish"
else
    echo >&2 "Not activating venv; missing dir: $venv_dir"
    echo >&2 "(Sorry, this script must be sourced from the repo directory.)"
    return 1
end

set -gx PYTHONPATH "$PYTHONPATH:$PWD/src"


# Set up auxilliary environment variables
# (See activate.sh for more details.)
set -gx PYTHONUNBUFFERED 1
set -gx _TYPER_STANDARD_TRACEBACK 1
