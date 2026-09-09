"""向量库集成测试：验证 Chroma 与内存实现接口一致

用固定的 one-hot 假向量驱动（不依赖真实 Embedding 模型），
因此不需要 sentence-transformers / torch，只依赖 chromadb 本身；
未安装 chromadb 时整个模块自动跳过，不影响主流程测试。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("chromadb", reason="未安装 chromadb，跳过 Chroma 集成测试")

from app.rag.chunker import Chunk  # noqa: E402
from app.rag.vector_store import ChromaStore, VectorStore, create_store  # noqa: E402

DIM = 8


def onehot(index: int, dim: int = DIM):
    """构造正交单位向量：第 i 维突出第 i 个样本，便于确定性断言"""
    vec = [0.0] * dim
    vec[index] = 1.0
    return vec


def make_chunks():
    return [
        Chunk("c1", "图书馆周一至周五 8:00 至 22:00 开放", {"source": "a.md", "category": "开放时间"}),
        Chunk("c2", "本科生可借 20 册，期限 30 天", {"source": "a.md", "category": "借阅规则"}),
        Chunk("c3", "3 楼自习室需预约，每次最长 4 小时", {"source": "b.md", "category": "座位"}),
    ]


def build_store(tmp_path, name="test_collection"):
    store = ChromaStore(persist_dir=str(tmp_path), collection_name=name)
    store.clear()
    chunks = make_chunks()
    store.add_batch(chunks, [onehot(i) for i in range(len(chunks))])
    return store


def test_chroma_search_returns_most_similar(tmp_path):
    store = build_store(tmp_path)
    assert store.size == 3

    results = store.search(onehot(0), None, top_k=3)
    assert results, "检索结果不应为空"
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > 0.99, f"完全相同的向量应得分为 1，实际 {results[0][1]}"

    results = store.search(onehot(2), None, top_k=1)
    assert results[0][0].chunk_id == "c3"


def test_chroma_metadata_filter(tmp_path):
    store = build_store(tmp_path)
    results = store.search(onehot(0), None, top_k=3, metadata_filter={"category": "借阅规则"})
    assert [cid for _, (chunk, _) in [(0, r) for r in results] for cid in [chunk.chunk_id]] == ["c2"]

    results = store.search(onehot(0), None, top_k=3, metadata_filter={"source": "b.md"})
    assert len(results) == 1 and results[0][0].chunk_id == "c3"


def test_chroma_persists_across_instances(tmp_path):
    """重启进程后索引仍在：验证 PersistentClient 真正落盘"""
    build_store(tmp_path)
    reopened = ChromaStore(persist_dir=str(tmp_path), collection_name="test_collection")
    assert reopened.size == 3, "Chroma 应在重新打开后保留索引"
    assert reopened.get_chunk("c1") is not None
    assert reopened.get_chunk("c1").content.startswith("图书馆")


def test_chroma_clear_and_rebuild(tmp_path):
    store = build_store(tmp_path)
    store.clear()
    assert store.size == 0
    assert store.search(onehot(0), None, top_k=3) == []


def test_interface_parity(tmp_path):
    """两种实现必须暴露同一组接口，保证上层可无感切换"""
    memory = VectorStore()
    chroma = ChromaStore(persist_dir=str(tmp_path), collection_name="parity")
    for store in (memory, chroma):
        for method in ("add", "search", "clear", "get_chunk", "to_json", "load_json"):
            assert hasattr(store, method), f"{type(store).__name__} 缺少方法 {method}"
        assert hasattr(store, "size")


def test_create_store_falls_back_without_chroma(monkeypatch):
    """chromadb 缺失时 create_store 应回退到内存实现而不是抛错"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "chromadb" or name.startswith("chromadb."):
            raise ImportError("模拟 chromadb 未安装")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    store = create_store("chroma", persist_dir=str(Path(tmp_path) / "x"))
    assert isinstance(store, VectorStore)
    assert not isinstance(store, ChromaStore)
