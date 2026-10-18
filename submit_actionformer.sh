#!/usr/bin/env bash
#SBATCH --job-name=intake-actionformer
#SBATCH --partition=luc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/seu_share2/home/213222097/IntakeDetection
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# 分区参照已有训练脚本；需要其他分区时用 sbatch -p <分区> 覆盖。
# 使用已验证的 Python 3.10 / PyTorch 2.1.2 环境，无需激活系统 Python。
task_project_dir="${INTAKE_PROJECT_DIR:-/seu_share2/home/213222097/IntakeDetection}"
task_python="${INTAKE_PYTHON:-/seu_share2/home/213222097/.conda/envs/meb_env/bin/python}"
task_config="${1:-experiment261016/intake_actionformer.yaml}"
cd "$task_project_dir"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

"$task_python" -m intake.run_job \
    --config "$task_config" \
    --run-id "${SLURM_JOB_ID:-local}" \
    --threads "$OMP_NUM_THREADS"
