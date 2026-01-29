import typer
from typing_extensions import Annotated

from . import __version__, identify_structure_and_annotate_gdx, plot_gdx


def version_callback(value: bool):
    if value:
        typer.echo(f"{__version__}")
        raise typer.Exit()


app = typer.Typer(
    add_completion=False,
)


@app.callback()
def common(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", callback=version_callback),
):
    pass


@app.command(
    help="Annotate a gdx file by specifiying a pattern and number of blocks."
)
def annotate(
    model_dump_file: Annotated[
        str, typer.Argument(help="Path to the gdx file.")
    ],
    pattern: Annotated[
        str,
        typer.Argument(
            help="Pattern to match set elements via regular expressions."
        ),
    ],
    blocks: Annotated[
        int, typer.Argument(help="Number of blocks for the annotation.")
    ],
    sort_order: Annotated[
        str,
        typer.Option(
            help="Sort order for the capture groups. Use e.g. [1] to sort blocks by the second capture group."
        ),
    ] = None,
    suffix: Annotated[
        str, typer.Option(help="Suffix for the annotated file.")
    ] = None,
) -> None:
    identify_structure_and_annotate_gdx(
        model_dump_file=model_dump_file,
        pattern_regex=pattern,
        blocks=blocks,
        sort_order=sort_order,
        suffix=suffix,
    )


@app.command(
    help="Detect structure in gdx and output a set of annotated gdx files."
)
def detect(
    model_dump_file: Annotated[
        str, typer.Argument(help="Path to the model dump gdx file.")
    ],
    model_dict_file: Annotated[
        str, typer.Argument(help="Path to the model dictionary gdx file.")
    ],
    all_symbols_file: Annotated[
        str, typer.Argument(help="Path to the model symbols gdx file.")
    ],
    suffix: Annotated[
        str, typer.Option(help="Suffix for the annotated file.")
    ] = None,
) -> None:
    identify_structure_and_annotate_gdx(
        model_dump_file=model_dump_file,
        model_dict_file=model_dict_file,
        all_symbols_file=all_symbols_file,
        suffix=suffix,
    )


@app.command(help="Plot the (annotion) structure of a gdx file.")
def plot(
    file: Annotated[str, typer.Argument(help="Path to the gdx file.")],
    figure: Annotated[
        str,
        typer.Argument(
            help="Filename if the figure should be saved to a file."
        ),
    ] = None,
    colormap: Annotated[
        str,
        typer.Option(help="Name of the colormap for the coloration of blocks."),
    ] = "viridis",
    plain: Annotated[
        bool, typer.Option(help="Plot the strucutre without any annotation.")
    ] = False,
) -> None:
    plot_gdx(file=file, figure=figure, colormap=colormap, plain=plain)


if __name__ == "__main__":
    app()
