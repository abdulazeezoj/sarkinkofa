import os
from typing import Optional
import typer
from sarkinkofa.tools import SARKINKofaFSWatcher

app = typer.Typer()


def start_watcher(input_folder: str, output_folder: str, verbose: bool = False):
    """
    Start the SARKINKofaFSWatcher.

    Args:
    - input_folder (str): Folder to watch.
    - output_folder (str): Folder to store output.
    - verbose (bool, optional): Enable verbose mode.
    """
    # Initialize watcher
    if verbose:
        typer.echo("[ INFO ] Initializing watcher...")

    image_watcher = SARKINKofaFSWatcher(
        input_folder=input_folder, output_folder=output_folder, verbose=verbose
    )

    if verbose:
        typer.echo("[ INFO ] Watcher initialized!")

    try:
        # Start watching
        if verbose:
            typer.echo("[ INFO ] Watching for images...")

        image_watcher.start_watching()

    except KeyboardInterrupt:
        if verbose:
            typer.echo("[ INFO ] Shutting down watcher...")

        image_watcher.stop_watching()

        if verbose:
            typer.echo("[ INFO ] Watcher stopped!")


@app.command()
def main(
    input_folder: str = typer.Argument(..., help="Folder to watch."),
    output_folder: str = typer.Argument(..., help="Folder to store output."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode."),
):
    """
    SARKINKofa File System Watcher CLI.
    """
    start_watcher(input_folder, output_folder, verbose)


if __name__ == "__main__":
    app()
