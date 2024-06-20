from pydantic import BaseModel, Field


class ImageData(BaseModel):
    image_id: int = Field(alias="id")
    license_id: int = Field(alias="license")
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str
