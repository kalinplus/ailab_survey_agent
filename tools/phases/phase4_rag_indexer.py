from tools.indexer.vector_store import VectorStore

def run(parsed_papers, embedding_client, path=None, extra_texts=None):
    vs = VectorStore(embedding_client, path=path)
    ids, texts, metas = [], [], []
    for p in parsed_papers.papers:
        if p.abstract:
            ids.append(f"{p.paper_id}_abs"); texts.append(p.abstract)
            metas.append({"paper_id": p.paper_id, "source_type": "abstract", "page": 0})
        for para in p.paragraphs:
            eid = f"{p.paper_id}_p{para.page}_{para.index}"
            ids.append(eid); texts.append(para.text)
            metas.append({"paper_id": p.paper_id, "source_type": "paragraph", "page": para.page})
    if extra_texts:
        ids += [e["id"] for e in extra_texts]; texts += [e["text"] for e in extra_texts]
        metas += [e["meta"] for e in extra_texts]
    if ids:
        vs.add(ids, texts, metas)
    return vs
