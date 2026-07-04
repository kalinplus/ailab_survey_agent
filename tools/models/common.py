import hashlib


def paper_id_from_seed(title: str, year: int) -> str:
    h = hashlib.sha256(f"{title}|{year}".encode()).hexdigest()[:12]
    return f"seed:{h}"


def evidence_id(paper_id: str, page: int, idx: int) -> str:
    return f"{paper_id}_p{page}_{idx}"


def figure_id(paper_id: str, num: int) -> str:
    return f"{paper_id}_fig{num}"


def table_id(paper_id: str, num: int) -> str:
    return f"{paper_id}_tbl{num}"


def category_id(n: int) -> str:
    return f"cat_{n:03d}"
