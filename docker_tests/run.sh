#!/bin/bash
set -e
NAME="CherenkovDeconvolution_jl" # container name

# find the name of the image (with or without prefix)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -f "${SCRIPT_DIR}/.IMAGE" ]; then
    IMAGE="$(cat "${SCRIPT_DIR}/.IMAGE")"
else
    echo "ERROR: Could not find any Docker image. Run 'make' first!"
    exit 1
fi

args="${@:1}"
docker run \
    --tty --interactive --rm \
    --volume /home/$USER:/mnt/home \
    --name "CherenkovDeconvolution_jl" \
    $IMAGE \
    $args # pass additional arguments to the container entrypoint
