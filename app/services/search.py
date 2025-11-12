from pathlib import Path
from typing import Dict
from whoosh import fields
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser


class _SearchIndex:
    def __init__(self) -> None:
        self._ix = None

    def initialize(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        schema = fields.Schema(id=fields.ID(stored=True, unique=True),
                               title=fields.TEXT(stored=True),
                               text=fields.TEXT(stored=True))
        try:
            self._ix = open_dir(str(dir_path))
        except Exception:
            self._ix = create_in(str(dir_path), schema)

    def add_or_update(self, id_: str, title: str, text: str = "") -> None:
        if not self._ix:
            return
        writer = self._ix.writer()
        writer.update_document(id=id_, title=title, text=text)
        writer.commit()

    def update_text(self, id_: str, text: str) -> None:
        if not self._ix:
            return
        # Need existing title; fetch via search by id
        with self._ix.searcher() as s:
            qp = MultifieldParser(["id"], schema=self._ix.schema)
            q = qp.parse(f"id:{id_}")
            results = s.search(q, limit=1)
            title = results[0]["title"] if results else ""
        self.add_or_update(id_, title, text)

    def run_query(self, query: str) -> Dict[str, str]:
        if not self._ix:
            return {}
        with self._ix.searcher() as searcher:
            parser = MultifieldParser(["title", "text"], schema=self._ix.schema)
            q = parser.parse(query)
            results = searcher.search(q, limit=20)
            return {hit["id"]: (hit.highlights("text") or hit["title"]) for hit in results}
    
    def delete(self, id_: str) -> None:
        if not self._ix:
            return
        writer = self._ix.writer()
        writer.delete_by_term("id", id_)
        writer.commit()


search_index = _SearchIndex()


def add_to_index(asset_id: str, title: str) -> None:
    search_index.add_or_update(asset_id, title)


def update_doc_text(asset_id: str, text: str) -> None:
    search_index.update_text(asset_id, text)


def search(query: str) -> Dict[str, str]:
    return search_index.run_query(query)


def remove_from_index(asset_id: str) -> None:
    search_index.delete(asset_id)




