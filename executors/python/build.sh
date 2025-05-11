#!/usr/bin/env bash

script_dir=$( dirname "$0" )
cd "$script_dir"


executor_image_tag=agnostics-python-executor
"${AGNOSTICS_CONTAINER_TOOL:-podman}" build -t "$executor_image_tag" .
