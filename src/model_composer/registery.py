import sys

import torch


class ModuleRegistry:
    """Register all module classes."""

    # Add torch modules to the registry
    REGISTRY = {
        "torch.nn." + name: obj
        for name, obj in torch.nn.__dict__.items()
        if isinstance(obj, type)
        and issubclass(obj, torch.nn.Module)
        and obj != torch.nn.Module
    }

    @classmethod
    def register(cls, subclass_name):
        """Register a subclass with the registry."""

        def decorator(subclass):
            if subclass_name in cls.REGISTRY:
                raise Warning(f"Overwriting {subclass_name}")
            cls.REGISTRY[subclass_name] = subclass
            return subclass

        return decorator

    @classmethod
    def build(cls, subclass_name, **kwargs):
        """Build a module from the registry.
        Args:
            subclass_name (str): Name of the subclass to build.
            **kwargs: Additional arguments to pass to the constructor.
        Returns:
            Module: An instance of the requested module.
        """
        if subclass_name not in cls.REGISTRY:
            raise ValueError(f"Unknown class {subclass_name}")
        return cls.REGISTRY[subclass_name](**kwargs)
