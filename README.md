# Fluxture

[![PyPI version](https://badge.fury.io/py/fluxture.svg)](https://badge.fury.io/py/fluxture)
[![Tests](https://github.com/trailofbits/fluxture/workflows/Tests/badge.svg)](https://github.com/trailofbits/fluxture/actions)
[![Slack Status](https://empireslacking.herokuapp.com/badge.svg)](https://empireslacking.herokuapp.com)

Fluxture is a lightweight crawler for peer-to-peer networks like Blockchains. It currently supports the latest version
of the Bitcoin protocol: 70015. It implements the minimum amount of the Bitcoin protocol necessary to collect geographic
and topological information.

## Quickstart

```commandline
pip3 install fluxture
```

Or, to install from source (_e.g._, for development):

```commandline
$ git clone https://github.com/trailofbits/fluxture
$ cd fluxture
$ pip3 install -e '.[dev]'
```

## Usage

To crawl the Bitcoin network, run:

```commandline
fluxture crawl bitcoin --database crawl.db
```

The crawl database is a SQLite database that can be reused between crawls.

## Geolocation

Fluxture uses the MaxMind GeoLite2 City database for geolocating nodes based upon their IP address. Various Fluxture
commands will either require a path to the database, or a MaxMind license key (which will be used to automatically
download the database). You can sign up for a free MaxMind license key,
[here](https://www.maxmind.com/en/geolite2/signup).

A KML file (which can be imported to Google Maps or Google Earth) can be generated from a crawl using:

```commandline
fluxture kml --group-by ip crawl.db output.kml
```

The geolocation database can be updated from MaxMind by running:

```commandline
fluxture update-geo-db
```

An existing crawl database can be re-analyzed for missing or updated geolocations (_e.g._, from an updated MaxMind database) by running:

```commandline
fluxture geolocate crawl.db
```

## Topological Analysis

Fluxture can calculate topological statistics about the centrality of a crawled network by running:

```commandline
fluxture topology crawl.db
```

## License and Acknowledgements

This research was developed by [Trail of Bits](https://www.trailofbits.com/) based upon work supported by DARPA under
Contract No. HR001120C0084.  Any opinions, findings and conclusions or recommendations expressed in this material are
those of the authors and do not necessarily reflect the views of the United States Government or DARPA.
It is licensed under the [Apache 2.0 license](LICENSE). © 2020–2021, Trail of Bits.
