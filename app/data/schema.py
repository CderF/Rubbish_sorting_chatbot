"""Pydantic models for the structured DB tables (documents/data_schema.md).

Mirrors `waste_items` and `collection_schedule` exactly. `id` is optional
because it isn't known before an INSERT (SQLite assigns it via AUTOINCREMENT).
"""

import unicodedata
from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


class WasteItem(BaseModel):
    """A row in `waste_items` (早見表パス)."""

    id: int | None = None
    item_name: str = Field(..., description="NFKC正規化・全角半角統一済みの検索キー")
    item_name_raw: str = Field(..., description="クロール時点のページ表記そのまま")
    disposal_method: str
    notes: str = ""
    source_url: str
    source_page_title: str | None = None
    crawled_at: datetime
    is_current: bool = True
    effective_from: date | None = None
    superseded_by: int | None = None

    @field_validator("item_name")
    @classmethod
    def _normalize_item_name(cls, v: str) -> str:
        return unicodedata.normalize("NFKC", v).strip()


class CollectionSchedule(BaseModel):
    """A row in `collection_schedule` (収集日パス)."""

    id: int | None = None
    ward: str
    town: str
    waste_type: str
    weekday: str = Field(..., description="複数曜日はカンマ区切り、例: '月,木'")
    source_url: str
    crawled_at: datetime
    is_current: bool = True
    effective_from: date | None = None
    superseded_by: int | None = None
