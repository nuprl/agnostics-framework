import importlib

from .common import commands


PROGLANGS: list[str] = []
BUILD_PROMPT_CMDS: dict[str, commands.CmdStandardMakePrompts] = {}
GENERATE_CMDS: dict[str, commands.CmdGenerate] = {}
VERIFY_CMDS: dict[str, commands.CmdVerifyAnswers] = {}
IMAGE_NAMES: dict[str, str] = { }
PREBUILT_IMAGE_NAMES: dict[str, str] = { }


def _register_proglang(name: str, *, module_name: str | None = None):
    module_name = module_name or name
    module = importlib.import_module(f".{module_name}", package=__package__)

    PROGLANGS.append(name)
    BUILD_PROMPT_CMDS[name] = module.command_make_prompts
    GENERATE_CMDS[name] = module.command_generate
    VERIFY_CMDS[name] = module.command_verify
    IMAGE_NAMES[name] = module.executor_image_name
    if hasattr(module, 'prebuilt_executor_image_name'):
        PREBUILT_IMAGE_NAMES[name] = module.prebuilt_executor_image_name


_register_proglang('python')
_register_proglang('cpp')
_register_proglang('c')
_register_proglang('java')
_register_proglang('lua')
_register_proglang('julia')
_register_proglang('r')
_register_proglang('ocaml')
_register_proglang('fortran')
