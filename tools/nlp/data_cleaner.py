import re
ABSTRACT_RE = re.compile(r"\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*", re.I)

class DataCleaner:
    def complete_abstract(self, paper: dict) -> str:
        abstract = paper.get("abstract", "") or ""
        if len(abstract) > 500:
            return abstract
        md = paper.get("md_text", "") or paper.get("full_text", "") or ""
        if not md:
            return abstract
        m = ABSTRACT_RE.search(md)
        if m:
            return md[m.end(): m.end() + 2000].strip()
        return md[:2000].strip()

    def clean_parsed(self, parsed: dict) -> dict:
        if not parsed.get("abstract"):
            parsed["abstract"] = self.complete_abstract(parsed)
        # flatten paragraphs from sections if missing
        if not parsed.get("paragraphs") and parsed.get("sections"):
            flat = []
            for sec in parsed["sections"]:
                for p in sec.get("paragraphs", []):
                    flat.append(p)
            parsed["paragraphs"] = flat
        return parsed
