import tarfile
import urllib.request
from ipaddress import IPv6Address as PythonIPv6, IPv4Address as PythonIPv4
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, Iterator, Optional, Union
from typing_extensions import Protocol

import geoip2.database
from fastkml import kml
from fastkml.geometry import LineString, Point
from fastkml.styles import IconStyle, LineStyle, Style

from .db import Model, Table
from .serialization import DateTime, IPv6Address


KML_NS = "{http://www.opengis.net/kml/2.2}"


class Geolocation(Model):
    ip: IPv6Address
    city: str
    lat: float
    lon: float
    timestamp: DateTime

    def to_placemark(self) -> kml.Placemark:
        icon = Style(KML_NS, styles=[IconStyle(KML_NS, id="ip")])
        p = kml.Placemark(
            KML_NS,
            str(int(self.rowid)),
            str(self.ip),
            f"{self.ip!s}: {self.city} ({self.lat}°N, {self.lon}°E) @ {self.timestamp!s}"
        )
        p.append_style(icon)
        p.geometry = Point(self.lon, self.lat)
        return p

    @classmethod
    def placemarks(cls, table: Table["Geolocation"]) -> Iterator[kml.Placemark]:
        for loc in table:
            yield loc.to_placemark()


def to_kml(
        table: Table[Geolocation],
        doc_id: str,
        doc_name: str,
        doc_description: str,
        edges: Callable[[Geolocation], Iterable[IPv6Address]] = lambda ip: ()
) -> kml.KML:
    k = kml.KML()
    d = kml.Document(KML_NS, doc_id, doc_name, doc_description)
    k.append(d)
    f = kml.Folder(KML_NS, "ips", "IPs", "Geolocalized IPs")
    d.append(f)
    for p in Geolocation.placemarks(table):
        f.append(p)
    edge_folder = kml.Folder(KML_NS, "topology", "Topology", "The network topology discovered in the crawl")
    d.append(edge_folder)
    edge_color = (0, 255, 0)
    edge_hex_color = "7f%02x%02x%02x" % tuple(reversed(edge_color))
    edge_style = Style(KML_NS, styles=[LineStyle(KML_NS, id="edge", color=edge_hex_color, width=3)])
    for geolocation in table:
        for edge in edges(geolocation):
            p = kml.Placemark(
                KML_NS,
                f"{geolocation.ip!s}->{edge!s}",
                f"{geolocation.ip!s}->{edge!s}",
                f"Edge between {geolocation.ip!s} and {edge!s}"
            )
            p.append_style(edge_style)
            edge_row = table.select(limit=1, ip=edge).fetchone()
            p.geometry = LineString([(geolocation.lon, geolocation.lat), (edge_row.lon, edge_row.lat)])
            edge_folder.append(p)
    return k


class Geolocator(Protocol):
    def locate(self, ip: Union[IPv6Address, str, PythonIPv4, PythonIPv6]) -> Geolocation:
        ...


class GeoIP2Error(RuntimeError):
    pass


class GeoIP2Locator:
    def __init__(self, city_db_path: Optional[str] = None, maxmind_license_key: Optional[str] = None):
        if city_db_path is None:
            city_db_path = Path.home() / ".config" / "fluxture" / "geolite2" / "GeoLite2-City.mmdb"
        else:
            city_db_path = Path(city_db_path)
        if not city_db_path.exists():
            if maxmind_license_key is None:
                raise GeoIP2Error("No MaxMind GeoLite2 database provided; need a `maxmind_license_key` to download it. "
                                  "Sign up for GeoLite2 for free here: https://www.maxmind.com/en/geolite2/signup "
                                  "then, after logging in, generate a license key under the Services menu.")
            db_dir = city_db_path.parent
            if not db_dir.exists():
                db_dir.mkdir(parents=True)
            tmpfile = NamedTemporaryFile(mode="wb", delete=False)
            try:
                with urllib.request.urlopen(r"https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-City&"
                                            f"license_key={maxmind_license_key}&suffix=tar.gz") as response:
                    # We have to write this to a temp file because the gzip decompression requires seekability
                    while True:
                        chunk = response.read(1024**2)
                        if not chunk:
                            break
                        tmpfile.write(chunk)
                tmpfile.close()
                with tarfile.open(tmpfile.name, mode="r:gz") as tar:
                    geolite_dir = None
                    for tarinfo in tar:
                        if tarinfo.isdir():
                            geolite_dir = tarinfo.name
                    if geolite_dir is None:
                        raise GeoIP2Error("Unexpected GeoLite2 database format")
                    tar.extractall(str(city_db_path.parent))
                    latest_dir = db_dir / "GeoLite2-City_latest"
                    latest_dir.symlink_to(db_dir / geolite_dir)
            finally:
                Path(tmpfile.name).unlink(missing_ok=True)
            city_db_path.symlink_to(latest_dir / "GeoLite2-City.mmdb")
        self.city_db_path: Path = city_db_path
        self._geoip: Optional[geoip2.database.Reader] = None
        self._entries: int = 0

    def __enter__(self):
        if self._entries == 0:
            assert self._geoip is None
            self._geoip = geoip2.database.Reader(str(self.city_db_path)).__enter__()
        self._entries += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._entries == 1:
            assert self._geoip is not None
            self._geoip.__exit__(exc_type, exc_val, exc_tb)
            self._geoip = None
        self._entries = max(0, self._entries - 1)

    def locate(self, ip: Union[IPv6Address, str, PythonIPv4, PythonIPv6]) -> Geolocation:
        with self:
            city = self._geoip.city(str(ip))
            return Geolocation(
                ip=IPv6Address(ip),
                city=city.city.name,
                lat=city.location.latitude,
                lon=city.location.longitude,
                timestamp=DateTime()
            )
