import os
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi

app = typer.Typer()


def main(
    repo_id: str,
    initial_commit: str,
    model_path: Path,
    new_model_name: str,
):
    assert model_path.is_dir(), f'--model-path should be a directory; got: {model_path}'
    assert list(model_path.glob('*.safetensors')), (
        f'--model-path should contain a .safetensors file; got: {model_path}'
    )

    api = HfApi(token=os.getenv('HF_TOKEN'))

    upload_res = api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type='model',
        revision=initial_commit,
        commit_message=new_model_name,
    )

    api.create_tag(
      repo_id=repo_id,
      revision=upload_res.oid,
      tag=new_model_name,
    )


@app.command()
def nuprl_public(
    model_path: Annotated[Path, typer.Option(help='Path to the model to upload')],
    new_model_name: Annotated[str, typer.Option(help='The name for the model after upload')],
):
    repo_id = 'nuprl/agnostics'
    initial_commit = '6f7b85d16248f5387c98be0473da66e93857c380'
    main(repo_id, initial_commit, model_path, new_model_name)


@app.command()
def nuprl_staging(
    model_path: Annotated[Path, typer.Option(help='Path to the model to upload')],
    new_model_name: Annotated[str, typer.Option(help='The name for the model after upload')],
):
    repo_id = 'nuprl-staging/agnostics'
    initial_commit = 'ffd54d6f54e46ac625805acccf00ff0b259d9831'
    main(repo_id, initial_commit, model_path, new_model_name)


if __name__ == '__main__':
    app()
