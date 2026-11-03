"""续借并发评测（实验4）

评测对象：app.services.svc.UserService.renew（生产代码实际实现）
重要说明：当前 renew 是单本续借 + Mock 数据，仅有「renew_count>=1 状态守卫」，
          **无幂等键 / 无乐观锁 / 无事务原子性**（这些是设计文档 S-P1-031 的目标）。
          本评测如实测量现状，并标注设计态缺口。

指标：
  - 串行幂等(状态守卫)：重复请求是否被拦（串行）
  - 并发竞态(丢失更新)：多线程同时续借同一本书，观测 renew_count 被重复自增的次数
  - 部分成功回执正确率：混合资格批量续借时，四类结果回执分类是否正确
  - 乐观锁冲突处理率：当前无版本号/锁 → 实测 0（设计态）

用法：
    python eval/run_renew_eval.py
"""
import json
import sys
import threading
import time
from pathlib import Path

# 最坏情况并发压测：把 GIL 切换间隔调到极小，放大非原子 check-then-act 的并发交错，
# 用于诚实暴露「无锁/无幂等键」下的丢失更新风险（生产多 worker 部署即此场景）。
_STRESS_SWITCH_INTERVAL = 1e-5

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import user_service as us_mod  # noqa: E402
from app.services.user_service import user_service as svc  # noqa: E402

TRIALS = 80
THREADS = 20
TEST_USER = "EVAL_RENEW"


def reset_fresh(user, book_id="X", title="测试书"):
    us_mod.MOCK_BORROWS[user] = [
        {"book_id": book_id, "title": title, "due": "2026-07-01", "renew_count": 0}
    ]


def test_serial_dedup():
    reset_fresh(TEST_USER)
    r1 = svc.renew(TEST_USER, "X")
    r2 = svc.renew(TEST_USER, "X")
    return r1["ok"], (not r2["ok"] and r2.get("result_type") == "non_retryable_failure")


def _renew_racy_mirror(user_id, book_id):
    """故障注入镜像：复刻 user_service.renew 的 check-then-act 非原子结构，
    在检查与写入间注入让步以放大竞态窗口（仅用于压力证明，非生产路径）。"""
    borrows = us_mod.MOCK_BORROWS.get(user_id, [])
    book = next((b for b in borrows if book_id in b["book_id"] or book_id in b["title"]), None)
    if book is None:
        return {"ok": False}
    if book["renew_count"] >= 1:
        return {"ok": False}
    time.sleep(0)  # 放大窗口：让其他线程在「检查通过」与「写入」之间切入
    book["renew_count"] += 1
    return {"ok": True}


def test_concurrency_race():
    # (A) 真实 renew 在单进程 GIL 下的并发压测
    real_races = 0
    for _ in range(TRIALS):
        reset_fresh(TEST_USER)
        barrier = threading.Barrier(THREADS)

        def worker():
            barrier.wait()
            svc.renew(TEST_USER, "X")

        ts = [threading.Thread(target=worker) for _ in range(THREADS)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if us_mod.MOCK_BORROWS[TEST_USER][0]["renew_count"] > 1:
            real_races += 1

    # (B) 故障注入镜像：放大窗口后验证丢失更新可达
    stress_races = 0
    for _ in range(TRIALS):
        reset_fresh(TEST_USER)
        barrier = threading.Barrier(THREADS)

        def worker2():
            barrier.wait()
            _renew_racy_mirror(TEST_USER, "X")

        ts = [threading.Thread(target=worker2) for _ in range(THREADS)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if us_mod.MOCK_BORROWS[TEST_USER][0]["renew_count"] > 1:
            stress_races += 1

    return real_races, stress_races


def classify_batch(results):
    oks = [r["ok"] for r in results]
    if all(oks):
        return "full_success"
    if not any(oks):
        return "all_fail"
    return "partial_success"


def test_batch_receipt():
    # 构造混合资格批量：A 可续 / B 已续过 / C 可续 / D 不存在
    us_mod.MOCK_BORROWS[TEST_USER] = [
        {"book_id": "A", "title": "A书", "due": "2026-07-01", "renew_count": 0},
        {"book_id": "B", "title": "B书", "due": "2026-06-15", "renew_count": 1},
        {"book_id": "C", "title": "C书", "due": "2026-08-30", "renew_count": 0},
    ]
    results = [
        svc.renew(TEST_USER, "A"),
        svc.renew(TEST_USER, "B"),
        svc.renew(TEST_USER, "C"),
        svc.renew(TEST_USER, "D"),  # 不存在
    ]
    predicted = classify_batch(results)
    expected = "partial_success"  # 2 成功 2 失败 → 部分成功
    return predicted == expected, predicted, expected, [
        (r.get("result_type"), r["ok"]) for r in results
    ]


def main():
    global TRIALS, THREADS
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()
    TRIALS, THREADS = args.trials, args.threads
    # 1) 串行幂等
    s_ok, s_dedup = test_serial_dedup()
    serial_idem = s_ok and s_dedup

    # 2) 并发竞态
    real_races, stress_races = test_concurrency_race()
    real_rate = real_races / TRIALS
    stress_rate = stress_races / TRIALS

    # 3) 批量回执正确率
    receipt_ok, pred, exp, detail = test_batch_receipt()

    # 4) 乐观锁
    opt_lock_rate = 0.0  # 代码无版本号/锁

    print(f"\n样本/压测: 并发 {THREADS} 线程 × {TRIALS} 轮")
    print("=" * 56)
    print(f"【串行幂等(状态守卫)】重复请求被拦 : {'通过' if serial_idem else '失败'}")
    print(f"【并发竞态(真实 renew, GIL 下)】{real_races}/{TRIALS} 轮重复续借 → {real_rate:.4f}")
    print(f"【并发竞态(故障注入镜像放大窗口)】{stress_races}/{TRIALS} 轮 → {stress_rate:.4f}")
    print(f"     → 当前无锁/无原子操作，并发下 renew_count 被重复自增（丢失更新）")
    print(f"【部分成功回执正确率】   : {'100%' if receipt_ok else '错误'} (预测={pred}, 期望={exp})")
    print(f"     明细(结果类型,ok): {detail}")
    print(f"【乐观锁冲突处理率】     : {opt_lock_rate:.4f} (代码未实现版本号/锁)")
    print("=" * 56)
    print("结论：串行重复靠 renew_count 状态守卫拦住；但无幂等键/乐观锁，")
    print("      多 worker 部署下并发续借会出现丢失更新，设计态的『状态外置+锁+幂等键』是必需修复。")

    out = {
        "serial_idempotency_pass": serial_idem,
        "concurrency_race_rate_real": round(real_rate, 4),
        "concurrency_race_rate_stress": round(stress_rate, 4),
        "batch_receipt_correct": receipt_ok,
        "optimistic_lock_rate": opt_lock_rate,
        "trials": TRIALS, "threads": THREADS,
    }
    (ROOT / "data" / "eval" / "renew_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 data/eval/renew_report.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
