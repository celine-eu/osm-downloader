#!/usr/bin/env python
from typing import Union
import yaml
import click
import osmnx
import geopandas as gpd
from pathlib import Path
import re
import time
import pandas as pd
import logging
import sys
import os
from dotenv import load_dotenv
import shutil

FORMATS = ["geojson", "parquet", "csv"]


def clean_cache(cache_dir: Path, refresh: bool, max_age_days: int):
    """Remove osmnx cache files if too old or if refresh is forced."""
    if refresh:
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return

    if max_age_days > 0:
        cutoff = time.time() - max_age_days * 86400
        for f in cache_dir.glob("*.json"):
            if f.stat().st_mtime < cutoff:
                f.unlink()


def setup_logger() -> logging.Logger:
    """Configure logger for CLI."""
    logger = logging.getLogger("osm_downloader")
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def sanitize_filename(name: str) -> str:
    """Make a safe filename component."""
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")


def fetch_data(area: str, entity: dict, logger: logging.Logger) -> gpd.GeoDataFrame:
    """Query OSMNX for one entity definition in a given area."""
    key: str = entity["key"]
    value: str = entity["value"]
    tags: dict[str, Union[str, bool, list[str]]] = (
        {key: True} if value == "*" else {key: value}
    )

    try:
        gdf = osmnx.features.features_from_place(area, tags=tags)
        return gdf
    except ValueError as e:
        if "No matching features" in str(e):
            logger.info(f"No features for {key}={value} in {area}")
        else:
            logger.error(f"Query failed for {key}={value} in {area}: {e}")
        return gpd.GeoDataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching {key}={value} in {area}: {e}")
        return gpd.GeoDataFrame()


def is_outdated(path: Path, max_age_days: int) -> bool:
    """Check if file is older than max_age_days."""
    if not path.exists():
        return True
    if max_age_days <= 0:
        return False
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days > max_age_days


@click.command()
@click.argument("config_file", required=False, type=click.Path(exists=True))
def osm_download(config_file: str):
    """Download OSM data from multiple areas/entities defined in YAML config."""
    logger = setup_logger()

    # Load .env file
    load_dotenv()

    # Determine config path: CLI > ENV > default
    config_path = (
        Path(config_file)
        if config_file
        else Path(os.getenv("CONFIG_PATH", "./osm_config.yaml"))
    )

    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"Failed to load config {config_path}: {e}")
        sys.exit(1)

    # Override output folder from DATA_DIR env if set
    out_fmt = cfg.get("output", {}).get("format", "geojson")
    out_dir = Path(os.getenv("DATA_DIR", cfg.get("output", {}).get("folder", "./data")))
    refresh = cfg.get("output", {}).get("refresh", False)
    max_age_days = cfg.get("output", {}).get("max_age_days", 120)

    osmnx.settings.use_cache = True
    # clean cache older than max_age_days
    clean_cache(Path(osmnx.settings.cache_folder), refresh, max_age_days)

    if out_fmt not in FORMATS:
        logger.critical(f"Unsupported output format: {out_fmt}")
        sys.exit(1)

    for area_cfg in cfg.get("areas", []):

        area_place = area_cfg.get("place", None)
        area_name = area_cfg.get("name", None)

        if not area_place:
            raise Exception("'place' is required")

        if not area_name:
            area_name = sanitize_filename(area_place)

        area_dir = out_dir / sanitize_filename(area_name)
        area_dir.mkdir(parents=True, exist_ok=True)

        groups = area_cfg.get("groups", {})
        for group_name, entities in groups.items():
            outfile = area_dir / f"{sanitize_filename(group_name)}.{out_fmt}"

            if (
                outfile.exists()
                and not refresh
                and not is_outdated(outfile, max_age_days)
            ):
                logger.info(f"Skipping {outfile} (up-to-date)")
                continue

            logger.info(f"Fetching group '{group_name}' in {area_place}")

            all_results = []
            for ent in entities:
                gdf = fetch_data(area_place, ent, logger)
                if not gdf.empty:
                    all_results.append(gdf)

            if not all_results:
                logger.warning(f"No data for group '{group_name}' in {area_place}")
                continue

            # Preserve the OSM index
            gdf_out = gpd.GeoDataFrame(pd.concat(all_results, ignore_index=False))

            # Reset index so element + id become columns
            gdf_out = gdf_out.reset_index()

            # Drop duplicates based on osm_id
            gdf_out = gdf_out.drop_duplicates(subset=["element", "id"])

            try:
                if out_fmt == "geojson":
                    gdf_out.to_file(outfile, driver="GeoJSON")
                elif out_fmt == "parquet":
                    gdf_out.to_parquet(outfile)

                logger.info(f"Saved {len(gdf_out)} records to {outfile}")
            except Exception as e:
                logger.error(f"Failed writing {outfile}: {e}")


if __name__ == "__main__":
    osm_download()
