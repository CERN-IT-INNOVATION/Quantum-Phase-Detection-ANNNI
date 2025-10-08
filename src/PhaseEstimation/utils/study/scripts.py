import subprocess
from pathlib import Path

from . import ENV_PATH, LOGS_PATH


def make_script(path, name, job_name, python_line):
    script_file = Path(path) / f"{name}.sh"
    with open(script_file, "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --time=10:00:00
#SBATCH --chdir={LOGS_PATH}

echo "Activating Environment"
source {ENV_PATH / "bin/activate"}

echo "Running script"
{python_line}
"""
        )


def run_script(path, name):
    script_file = Path(path) / f"{name}.sh"
    subprocess.run(["sbatch", str(script_file)])
