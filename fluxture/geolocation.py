import tarfile
import urllib.request
from ipaddress import IPv4Address as PythonIPv4
from ipaddress import IPv6Address as PythonIPv6
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, Optional, Tuple, Union

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import geoip2.database
import great_circle_calculator.great_circle_calculator as gcc
from geoip2.errors import AddressNotFoundError

from .db import Model
from .serialization import DateTime, IPv6Address


class Location:
    lat: float
    lon: float

    def path_to(
        self, destination: "Location", intermediate_points: int = 20
    ) -> Iterator[Tuple[int, int]]:
        p1, p2 = (self.lon, self.lat), (destination.lon, destination.lat)
        yield self.lon, self.lat
        for i in range(intermediate_points):
            try:
                yield gcc.intermediate_point(
                    p1, p2, (i + 1) / (intermediate_points + 2)
                )
            except ZeroDivisionError:
                # this probably means p1 == p2
                yield self.lon, self.lat
        yield destination.lon, destination.lat

    def distance_to(self, destination: "Location", unit: str = "meters"):
        return gcc.distance_between_points(
            (self.lon, self.lat), (destination.lon, destination.lat), unit=unit
        )


class Geolocation(Model, Location):
    ip: IPv6Address
    city: str
    country_code: str
    continent_code: str
    lat: float
    lon: float
    timestamp: DateTime

    def __hash__(self):
        return hash(self.ip)


class Geolocator(Protocol):
    def locate(
        self, ip: Union[IPv6Address, str, PythonIPv4, PythonIPv6]
    ) -> Geolocation:
        ...


class GeoIP2Error(RuntimeError):
    pass


def download_maxmind_db(
    maxmind_license_key: str, city_db_path: Optional[str] = None, overwrite: bool = True
) -> str:
    """
    Downloads the latest MaxMind GeoLite2 database returning the path to which it was saved.

    If the path is omitted, the default path is used and returned.

    """
    if city_db_path is None:
        city_db_path = (
            Path.home() / ".config" / "fluxture" / "geolite2" / "GeoLite2-City.mmdb"
        )
    else:
        city_db_path = Path(city_db_path)
    if overwrite or not city_db_path.exists():
        if maxmind_license_key is None:
            raise GeoIP2Error(
                "No MaxMind GeoLite2 database provided; need a `maxmind_license_key` to download it. "
                "Sign up for GeoLite2 for free here: https://www.maxmind.com/en/geolite2/signup "
                "then, after logging in, generate a license key under the Services menu."
            )
        db_dir = city_db_path.parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True)
        tmpfile = NamedTemporaryFile(mode="wb", delete=False)
        try:
            with urllib.request.urlopen(
                r"https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-City&"
                f"license_key={maxmind_license_key}&suffix=tar.gz"
            ) as response:
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
                
                import os
                
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner) 
                    
                
                safe_extract(tar, str(city_db_path.parent))
                latest_dir = db_dir / "GeoLite2-City_latest"
                latest_dir.unlink(missing_ok=True)
                latest_dir.symlink_to(db_dir / geolite_dir)
        finally:
            Path(tmpfile.name).unlink(missing_ok=True)
        city_db_path.unlink(missing_ok=True)
        city_db_path.symlink_to(latest_dir / "GeoLite2-City.mmdb")
    return str(city_db_path)


class GeoIP2Locator:
    def __init__(
        self,
        city_db_path: Optional[str] = None,
        maxmind_license_key: Optional[str] = None,
    ):
        self.city_db_path: Path = download_maxmind_db(
            maxmind_license_key, city_db_path, overwrite=False
        )  # type: ignore
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

    def locate(
        self, ip: Union[IPv6Address, str, PythonIPv4, PythonIPv6]
    ) -> Geolocation:
        with self:
            ipv6 = IPv6Address(ip)
            city = self._geoip.city(str(ipv6))
            if city.location.latitude is None or city.location.longitude is None:
                raise AddressNotFoundError(str(ip))
            return Geolocation(
                ip=ipv6,
                city=city.city.name,
                country_code=city.country.iso_code,
                continent_code=city.continent.code,
                lat=city.location.latitude,
                lon=city.location.longitude,
                timestamp=DateTime(),
            )
