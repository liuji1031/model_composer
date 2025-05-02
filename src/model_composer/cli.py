import os
import yaml
import typer
from typing import Optional
from pathlib import Path

app = typer.Typer()


def load_yaml(file_path: str) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        typer.echo(f"Error loading {file_path}: {e}")
        raise typer.Exit(1)


def save_yaml(data: dict, file_path: str) -> None:
    """Save a dictionary as a YAML file."""
    try:
        with open(file_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    except Exception as e:
        typer.echo(f"Error saving {file_path}: {e}")
        raise typer.Exit(1)


def replace_hyper_params(template: dict, params: dict, overwrite: bool) -> dict:
    """Replace hyperparameters in the template with the provided params."""
    result = template.copy()
    if "hyper_params" not in result:
        result["hyper_params"] = params
    else:
        for name, value in params.items():
            if name in result["hyper_params"]:
                if overwrite:
                    result["hyper_params"][name] = value
                else:
                    typer.echo(
                        (
                            f"Warning: {name} already exists in hyper_params."
                            " Use --overwrite to replace."
                        )
                    )
            else:
                result["hyper_params"][name] = value
    return result


@app.command()
def generate(
    template_file: str = typer.Argument(
        ..., 
        help="Path to the template YAML file",
        dir_okay=False,
        resolve_path=True,
        exists=True,
        readable=True,
    ),
    params_file: str = typer.Argument(
        ..., help="Path to the hyperparameters YAML file",
        dir_okay=False,
        resolve_path=True,
        exists=True,
        readable=True,
    ),
    overwrite: bool = typer.Option(
        True, help="Overwrite existing hyperparameters"
    ),
):
    """
    Generate configuration files by replacing hyperparameters in a template
    with values from a hyperparameters file.
    """
    # Load the template and hyperparameters files
    template_data = load_yaml(template_file)
    params_data = load_yaml(params_file)

    # Get the output directory (same as the params file)
    output_dir = os.path.dirname(params_file)

    # Generate a config file for each field in the params file
    for module_name, params in params_data.items():
        # Create the output file path
        output_file = os.path.join(output_dir, f"{module_name}.yaml")
        
        # Replace the hyperparameters in the template
        config_data = replace_hyper_params(template_data, params, overwrite)
        config_data["name"] = module_name
        # Save the new config file
        save_yaml(config_data, output_file)
        typer.echo(f"Generated {output_file}")


if __name__ == "__main__":
    app()
