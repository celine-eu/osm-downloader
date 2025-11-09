from typing import Union
import yaml
import osmnx
import geopandas as gpd
from pathlib import Path
import re
import time
import pandas as pd
from osm_downloader.logger import get_logger
import logging
import sys
import os
from dotenv import load_dotenv
import shutil

from osm_downloader.storage import get_storage  # ← integrate storage layer

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


def osm_download(config_file: str | None):
    """Download OSM data from multiple areas/entities defined in YAML config."""
    logger = get_logger()

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

    # Initialize storage backend (FS or S3)
    storage = get_storage(cfg)

    # Override output folder from DATA_DIR env if set
    out_fmt = cfg.get("output", {}).get("format", "geojson")
    out_folder = cfg.get("output", {}).get("folder", "./data")
    out_dir = Path(os.getenv("DATA_DIR", out_folder))
    refresh = cfg.get("output", {}).get("refresh", False)
    max_age_days = cfg.get("output", {}).get("max_age_days", 120)

    osmnx.settings.use_cache = True
    clean_cache(Path(osmnx.settings.cache_folder), refresh, max_age_days)

    if out_fmt not in FORMATS:
        logger.critical(f"Unsupported output format: {out_fmt}")
        sys.exit(1)

    for area_cfg in cfg.get("areas", []):
        area_place = area_cfg.get("place")
        area_name = area_cfg.get("name") or sanitize_filename(area_place)

        if not area_place:
            raise Exception("'place' is required")

        groups = area_cfg.get("groups", {})
        for group_name, entities in groups.items():
            # Construct storage key
            key = f"{sanitize_filename(area_name)}/{sanitize_filename(group_name)}.{out_fmt}"

            # Local temp path (for write/read ops)
            local_tmp = Path("/tmp") / key
            local_tmp.parent.mkdir(parents=True, exist_ok=True)

            # Check existence through storage
            if storage.exists(key) and not refresh:
                local_path = storage.get_path(key)
                if not is_outdated(Path(local_path), max_age_days):
                    logger.info(f"Skipping {key} (up-to-date in storage)")
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

            gdf_out = gpd.GeoDataFrame(pd.concat(all_results, ignore_index=False))
            gdf_out = gdf_out.reset_index().drop_duplicates(subset=["element", "id"])

            try:
                # Write to local tmp
                if out_fmt == "geojson":
                    gdf_out.to_file(local_tmp, driver="GeoJSON")
                elif out_fmt == "parquet":
                    gdf_out.to_parquet(local_tmp)

                # Upload to storage (or move for FS)
                storage.save(str(local_tmp), key)
                logger.info(
                    f"Saved {len(gdf_out)} records to {key} via {storage.__class__.__name__}"
                )
            except Exception as e:
                logger.error(f"Failed writing {key}: {e}")
