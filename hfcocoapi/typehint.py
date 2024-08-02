import pathlib
from typing import Annotated, Any, Dict, List, Literal, Tuple, Union

from PIL.Image import Image
from typing_extensions import TypedDict

JsonDict = Dict[str, Any]
ImageId = int
AnnotationId = int
LicenseId = int
CategoryId = int
Bbox = Tuple[float, float, float, float]

MscocoSplits = Literal["train", "val", "test"]

PilImage = Annotated[Image, "Pillow Image"]

PathLike = Union[str, pathlib.Path]


class CategoryDict(TypedDict):
    category_id: CategoryId
    name: str
    supercategory: str


class LicenseDict(TypedDict):
    license_id: LicenseId
    name: str
    url: str


class UncompressedRLE(TypedDict):
    counts: List[int]
    size: Tuple[int, int]


class CompressedRLE(TypedDict):
    counts: bytes
    size: Tuple[int, int]


class BaseExample(TypedDict):
    image_id: ImageId
    image: PilImage
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str
    license_id: LicenseId
    license: LicenseDict
