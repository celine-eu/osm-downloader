#!/usr/bin/env python
import click

from .downloader import osm_download


@click.command()
@click.argument("config_file", required=False, type=click.Path(exists=True))
def main(config_file: str | None):
    osm_download(config_file)


if __name__ == "__main__":
    main()
