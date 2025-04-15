import torch
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig, ListConfig
from model_composer.registery import ModuleRegistry
from model_composer.util import read_config


@ModuleRegistry.register("ComposableModel")
class ComposableModel(torch.nn.Module):
    """The ModelComposer class composes the computation required of the model
    by parsing the configuration yaml file. The yaml file defines the input and
    output of each module, and the connections between them.
    """

    def __init__(self, **kwargs):
        """Initialize the model."""
        super().__init__()
        self.name = kwargs.get("name", "model")
        self._inp_src = {}
        self._inp_num = {}
        self._out_varname = {}
        self._out_num = {}
        self._module_output = {}
        self._des = {}
        self._return_dict = False
        self._skip_build = kwargs.get("skip_build", False)
        assert "modules" in kwargs, (
            "'modules' field not found in the config file"
        )
        self._build(kwargs["modules"])
        self._op_seq = self._forward_dry_run()

    def _get_varname(self, module_name, iout):
        """Construct the output variable name for a module."""
        return f"{module_name}.{iout}"

    def _build(self, module_dict: dict | DictConfig):
        """Build the model based on config file.

        Args:
            module_list (list or ListConfig): list of configuration dictionary for each module
        """
        all_names = list(module_dict.keys())
        assert "entry" in all_names, (
            "'entry' module not found in the config file"
        )
        assert "exit" in all_names, "'exit' module not found in the config file"

        for module_name, module_config in module_dict.items():
            self._build_module(module_name, module_config)

    def _parse_des_varname(self, inp_src: str) -> tuple[str, int]:
        """Parse the destination string to get the module name and input port number."""
        # e.g. "module_name.input.0" -> ("module_name", 0)
        parsed_src = inp_src.split(".")
        return parsed_src[0], int(parsed_src[-1])

    def _validate_varname(self, inp_src: str) -> str:
        """Validate the input source string to ensure it is in the correct format."""
        parsed_src = inp_src.split(".")
        if len(parsed_src) == 1:
            pass
        elif len(parsed_src) == 2:
            # check if the second element is a valid integer
            if not parsed_src[1].isdigit():
                raise ValueError(
                    f"Invalid input source format: {inp_src}, integer expected after '.'"
                )
        if len(parsed_src) > 2:
            raise ValueError(f"Invalid input source format: {inp_src}")
        return inp_src

    def _build_module(self, module_name:str, module_config:dict|DictConfig):
        """Build a single module based on config file.

        Args:
            module_config (dict): configuration dictionary for a single module
        """
        print(f"Building module {module_name}")
        self._inp_src[module_name] = {}

        if module_name not in ["entry", "exit"] and not self._skip_build:
            # build the module
            assert "cls" in module_config, (
                "'cls' field not found in the module config file"
            )
            cls_name = module_config["cls"]
            if "config" in module_config:
                cfg = module_config["config"]
                if isinstance(cfg, (dict, DictConfig)):
                    module = ModuleRegistry.build(cls_name, **cfg)
                elif isinstance(cfg, str):
                    # if config is a path, read the config file
                    module = ModuleRegistry.build(cls_name, **read_config(cfg))
                else:
                    raise ValueError(
                        f"Unknown config type {type(cfg)} for module {module_name}"
                    )
            else:
                # with default setting
                module = ModuleRegistry.build(cls_name)
            self.register_module(module_name, module)

        # parse and validate input source variables
        if module_name in ["entry", "exit"]:
            inp_src = module_config # directly specified in the config
        else:
            inp_src = module_config["inp_src"]
        if isinstance(inp_src, (list, ListConfig)):
            self._inp_src[module_name] = []
            for src in inp_src:
                self._inp_src[module_name].append(self._validate_varname(src))
        elif isinstance(inp_src, (dict, DictConfig)):
            self._inp_src[module_name] = {}
            for des, src in inp_src.items():
                self._inp_src[module_name][des] = self._validate_varname(src)

        self._inp_num[module_name] = len(self._inp_src[module_name])

        if module_name not in ["entry", "exit"]:
            self._out_num[module_name] = module_config.get("out_num", 1)
        else:
            # entry and exit modules have no output
            self._out_num[module_name] = len(self._inp_src[module_name])

        # construct all output arg names
        if module_name == "entry":
            # entry module takes input from positional arguments
            self._out_varname[module_name] = self._inp_src[module_name]
        else:
            if isinstance(self._inp_src[module_name], (dict, DictConfig)):
                # exit module takes input from a dictionary
                self._out_varname[module_name] = list(
                    self._inp_src[module_name].keys()
                )
            else:
                if self._out_num[module_name] == 1:
                    # single output, use the name of the module
                    self._out_varname[module_name] = [module_name]
                else:
                    # multiple outputs, use the module name and index
                    # e.g. module_name.0, module_name.1, ...
                    self._out_varname[module_name] = [
                        self._get_varname(module_name, i)
                        for i in range(self._out_num[module_name])
                    ]

        if module_name not in ["entry", "exit"]:
            for iarg, src in enumerate(self._inp_src[module_name]):
                # create the list of destinations for each source
                if src not in self._des:
                    self._des[src] = []
                self._des[src].append(f"{module_name}.input.{iarg}")

        if module_name == "exit" and isinstance(
            self._inp_src["exit"], (dict, DictConfig)
        ):
            self._return_dict = True

    def _forward_dry_run(self):
        """Figure out the sequence of operations by simulating forward pass."""
        op_seq = []
        # create counter of input number ready for each module
        inp_cntr = {}
        for module_name in self._inp_src.keys():
            inp_cntr[module_name] = 0

        # queue of modules to be processed, start with entry
        q = ["entry"]
        while len(q) > 0:
            module_name = q.pop(0)
            out_varname = self._out_varname[module_name]

            for varname in out_varname:
                if varname not in self._des:
                    continue
                # update the input counter for destination modules that depend
                # on this output
                for des in self._des[varname]:
                    des_module_name, des_port = self._parse_des_varname(des)
                    inp_cntr[des_module_name] += 1
                    if (
                        inp_cntr[des_module_name]
                        == self._inp_num[des_module_name]
                    ):
                        # all inputs are ready, add module to queue
                        q.append(des_module_name)

            if module_name not in ["entry", "exit"]:
                op_seq.append(module_name)  # add module to the sequence

        return op_seq

    def forward(self, *args):
        """Forward pass of the model.

        Args:
            inp1 (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        module_output = {}
        assert len(args) == self._inp_num["entry"], (
            f"Number of input arguments {len(args)} does not match the number of "
            f"input sources {self._inp_num['entry']} for entry module"
        )
        for iarg, arg_name in enumerate(self._out_varname["entry"]):
            # entry module takes input from positional arguments
            module_output[arg_name] = args[iarg]

        for module_name in self._op_seq:
            # construct input dictionary for the module
            inp = [None for _ in range(self._inp_num[module_name])]
            for iarg, src in enumerate(self._inp_src[module_name]):
                inp[iarg] = module_output[src]
            try:
                out = self._modules[module_name](*inp)
            except Exception as e:
                logger.error(f"Error in module {module_name}: {e}")
                raise e

            # out is either a single tensor or a tuple of tensors
            if not isinstance(out, tuple):
                out = (out,)

            assert len(out) == self._out_num[module_name], (
                f"Number of output arguments {len(out)} does not match the number of "
                f"output sources {self._out_num[module_name]} defined for module {module_name}"
            )
            for varname, out_val in zip(self._out_varname[module_name], out):
                module_output[varname] = out_val

        if self._return_dict:
            final_output = {}
            for des, src in self._inp_src["exit"].items():
                # inp_src is a dict
                final_output[des] = module_output[src]
            return final_output
        else:
            final_output = []
            for src in self._inp_src["exit"]:
                final_output.append(module_output[src])
            if len(final_output) == 1:
                return final_output[0]
            else:
                return tuple(final_output)
