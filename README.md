# openstreetmap-downloader

Downloads OpenStreetMap data by configurations

## Setup

```bash
# Install dependencies
uv sync
```

## Usage

```bash
# Run importer
uv run -m osm_downloader.main
```

Usage from python

```python
from osm_downloader import osm_download

osm_download()
```

## Configuration

Check [osm_config.yaml](./osm_config.yaml) for a configuration example.

This examples creates a file `./data/borgo_vals/parking.geojson` with all records matching the properties in `groups.parking` (parking areas and EV charging stations).

```yaml
areas:
  - name: borgo_vals
    place: "Borgo Valsugana, Trentino"
    groups:
      parking:
        - key: "amenity"
          value: "parking"
        - key: "amenity"
          value: "parking_space"
        - key: "amenity"
          value: "charging_station"
```

A `.env` file can be used to control those variables

```ini
# base path where to save downloaded data
DATA_DIR=./data

# path to config file
CONFIG_PATH=./osm_config.yaml
```

## License

Licensed under `Apache-2.0` see [LICENSE](./LICENSE)