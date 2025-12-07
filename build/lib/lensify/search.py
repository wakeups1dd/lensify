import os, re, unicodedata, json
from pathlib import Path

# optional imports
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    CrossEncoder = None
    _HAS_ST = False

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

try:
    import PyPDF2
    _HAS_PYPDF2 = True
except Exception:
    PyPDF2 = None
    _HAS_PYPDF2 = False

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_SKLEARN = True
except Exception:
    TfidfVectorizer = None
    _HAS_SKLEARN = False

import numpy as np
from tqdm import tqdm

# hyperparams
CHUNK_MAX_TOKENS = 500  # approximate characters/words, tuned heuristically
CHUNK_OVERLAP = 0.2

def normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sentence_chunks(text, max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP):
    if not _HAS_NLTK:
        # simple fallback: split by newline paragraphs
        parts = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        for p in parts:
            if len(p) <= max_tokens:
                chunks.append(p)
            else:
                # naive fixed-size chunks
                i=0
                while i < len(p):
                    chunks.append(p[i:i+max_tokens])
                    i += int(max_tokens*(1-overlap))
        return chunks
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text)
    
def sentence_chunks(text, max_len=500):
    # Basic sentence splitting without NLTK
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < max_len:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks

def read_file(path: Path):
    suffix = path.suffix.lower()
    if suffix in ['.md', '.txt']:
        return path.read_text(encoding='utf-8', errors='ignore')
    if suffix == '.pdf':
        if not _HAS_PYPDF2:
            return ''
        try:
            reader = PyPDF2.PdfReader(str(path))
            pages = []
            for pnum, page in enumerate(reader.pages):
                txt = page.extract_text() or ''
                pages.append(f'[page:{pnum+1}]\n' + txt)
            return '\n\n'.join(pages)
        except Exception:
            return ''
    # fallback
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''

class Index:
    def __init__(self):
        self.docs = []  # dicts: id, path, chunk, meta, mtime
        self.embeddings = None
        self._faiss_index = None
        self.dim = None

    def save(self, p):
        data = {'docs': self.docs, 'embeddings': self.embeddings}
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, p):
        inst = cls()
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        inst.docs = data.get('docs', [])
        inst.embeddings = data.get('embeddings', None)
        if inst.embeddings is not None:
            arr = np.array(inst.embeddings, dtype='float32')
            inst.dim = arr.shape[1]
            inst._build_faiss_from_array(arr)
        return inst

    def _build_faiss_from_array(self, arr):
        if not _HAS_FAISS:
            self._np_emb = arr
            norms = np.linalg.norm(self._np_emb, axis=1, keepdims=True)
            norms[norms==0]=1.0
            self._np_emb = (self._np_emb / norms).astype('float32')
            self.dim = self._np_emb.shape[1]
            return
        dim = arr.shape[1]
        self.dim = dim
        # HNSW index
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        faiss.normalize_L2(arr)
        index.add(arr)
        self._faiss_index = index

class Search:
    def __init__(self, root_dir, index_path=None, embed_model=None, reranker_model=None):
        self.root = Path(root_dir).expanduser()
        self.index_path = Path(index_path) if index_path else (self.root / '.lensify_index.json')
        self.index = Index()
        self.embed_model_name = embed_model or ( 'BAAI/bge-large-en-v1.5' if _HAS_ST else None )
        self.reranker_name = reranker_model or ('cross-encoder/ms-marco-MiniLM-L-6-v2' if _HAS_ST else None)
        self.embedder = None
        self.reranker = None
        self._load_index_if_exists()

    def _load_index_if_exists(self):
        if self.index_path.exists():
            try:
                self.index = Index.load(str(self.index_path))
            except Exception:
                self.index = Index()

    def collect_files(self):
        exts = ['.md', '.txt', '.pdf']
        files = []
        for p in self.root.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                if any(part.startswith('.') for part in p.parts):
                    continue
                files.append(p)
        return files

    def build(self, force=False, show_progress=False):
        files = self.collect_files()
        changed = []
        for p in files:
            m = p.stat().st_mtime
            found = next((d for d in self.index.docs if d['path']==str(p)), None)
            if not found or found.get('mtime')!=m or force:
                changed.append(p)
        existing = {str(p) for p in files}
        self.index.docs = [d for d in self.index.docs if d['path'] in existing]

        for p in (tqdm(changed) if show_progress else changed):
            raw = read_file(p)
            raw = normalize_text(raw)
            chunks = sentence_chunks(raw)
            for i,c in enumerate(chunks):
                doc = {'id': f"{p}-{i}", 'path': str(p), 'chunk': c, 'meta':{}, 'mtime': p.stat().st_mtime}
                # page metadata
                if c.strip().startswith('[page:'):
                    try:
                        first, rest = c.split('\n',1)
                        page = int(first.split(':')[1].strip(']'))
                        doc['meta']['page'] = page
                        doc['chunk'] = rest
                    except Exception:
                        pass
                self.index.docs.append(doc)

        texts = [d['chunk'] for d in self.index.docs]
        if len(texts)==0:
            self.index.embeddings = []
            try:
                self.index.save(str(self.index_path))
            except Exception:
                pass
            return 0

        # embeddings
        if self.embed_model_name and _HAS_ST:
            try:
                self.embedder = SentenceTransformer(self.embed_model_name) if self.embedder is None else self.embedder
                embs = self.embedder.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
                # normalize
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms==0]=1.0
                embs = (embs / norms).astype('float32')
                self.index.embeddings = embs.tolist()
                self.index._build_faiss_from_array(embs)
            except Exception as e:
                # fallback to TF-IDF
                self._tfidf_fallback(texts)
        else:
            self._tfidf_fallback(texts)

        try:
            self.index.save(str(self.index_path))
        except Exception:
            pass
        return len(self.index.docs)

    def _tfidf_fallback(self, texts):
        if _HAS_SKLEARN:
            vec = TfidfVectorizer(stop_words='english', max_features=4096)
            X = vec.fit_transform(texts)
            arr = X.toarray().astype('float32')
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms==0]=1.0
            arr = (arr / norms)
            self.index.embeddings = arr.tolist()
            self.index._build_faiss_from_array(arr)
            self.vectorizer = vec
        else:
            # naive char average fallback
            arr = []
            for t in texts:
                if len(t)==0:
                    arr.append([0.0])
                else:
                    a = np.array([ord(ch) for ch in t[:512]], dtype=float)
                    arr.append([float(a.mean())])
            arr = np.array(arr, dtype='float32')
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms==0]=1.0
            arr = arr / norms
            self.index.embeddings = arr.tolist()
            self.index._build_faiss_from_array(arr)

    def _ensure_models(self):
        if self.embedder is None and self.embed_model_name and _HAS_ST:
            try:
                self.embedder = SentenceTransformer(self.embed_model_name)
            except Exception:
                self.embedder = None
        if self.reranker is None and self.reranker_name and _HAS_ST:
            try:
                self.reranker = CrossEncoder(self.reranker_name)
            except Exception:
                self.reranker = None

    def _embed_query(self, query):
        self._ensure_models()
        if self.embedder:
            q = self.embedder.encode([query], convert_to_numpy=True)[0]
            q = q.astype('float32')
            n = np.linalg.norm(q)
            if n>0:
                q = q / n
            return q
        if hasattr(self.index, '_np_emb'):
            # use simple average char feature
            arr = np.array([ord(ch) for ch in query[:512]], dtype=float)
            return np.array([arr.mean()], dtype='float32')
        return None

    def _search_faiss(self, qvec, top_k):
        if qvec is None:
            return []
        if _HAS_FAISS and getattr(self.index, '_faiss_index', None) is not None:
            q = qvec.reshape(1,-1).astype('float32')
            faiss.normalize_L2(q)
            D, I = self.index._faiss_index.search(q, top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0: continue
                doc = self.index.docs[int(idx)]
                results.append({'score': float(score), 'doc': doc})
            return results
        else:
            # numpy search
            emb = getattr(self.index, '_np_emb', None)
            if emb is None:
                return []
            qn = qvec / (np.linalg.norm(qvec)+1e-12)
            sims = np.dot(emb, qn)
            order = np.argsort(-sims)[:top_k]
            results = []
            for idx in order:
                results.append({'score': float(sims[idx]), 'doc': self.index.docs[int(idx)]})
            return results

    def _rerank(self, query, candidates):
        # candidates: list of {'score', 'doc'}
        self._ensure_models()
        if not self.reranker:
            return candidates
        pairs = [(query, c['doc']['chunk']) for c in candidates]
        scores = self.reranker.predict(pairs)
        for i,s in enumerate(scores):
            candidates[i]['rerank_score'] = float(s)
        candidates = sorted(candidates, key=lambda x: x.get('rerank_score', x['score']), reverse=True)
        return candidates

    def expand_query(self, query):
        # simple expansion heuristics
        return [
            query,
            f"explain {query}",
            f"what is {query}",
            f"examples of {query}"
        ]

    def query(self, query, top_k=6, rerank_k=20, score_threshold=0.0):
        if not self.index.docs:
            return []
        # expand & average embeddings
        queries = self.expand_query(query)
        qvecs = []
        for q in queries:
            v = self._embed_query(q)
            if v is not None:
                qvecs.append(v)
        if not qvecs:
            # fallback to single
            qvec = self._embed_query(query)
        else:
            qvec = np.mean(np.stack(qvecs, axis=0), axis=0)
        # search
        candidates = self._search_faiss(qvec, rerank_k)
        # rerank
        reranked = self._rerank(query, candidates)
        out = []
        for r in reranked[:top_k]:
            sc = r.get('rerank_score', r['score'])
            if sc < score_threshold:
                continue
            doc = r['doc']
            out.append({'score': float(sc), 'path': doc['path'], 'chunk': doc['chunk'], 'meta': doc.get('meta',{})})
        return out

    def stats(self):
        return {
            'num_documents': len(set(d['path'] for d in self.index.docs)),
            'num_chunks': len(self.index.docs),
            'index_path': str(self.index_path),
            'embed_model': self.embed_model_name,
            'reranker_model': self.reranker_name,
        }

    def export(self, outpath):
        data = {'docs': self.index.docs, 'embeddings': self.index.embeddings}
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

