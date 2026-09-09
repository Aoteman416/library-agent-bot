"""意图分类器冲突场景测试用例"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from app.services.chat_service import IntentClassifier

c = IntentClassifier()

# ========== 冲突场景测试 ==========
test_cases = [
    # === 场景1：业务 vs FAQ 重叠，FAQ 关键词长更占优 ===
    ('借书规则是什么', 'policy_faq', 'faq', '借书规则=5 vs 借阅=3，FAQ应胜出'),
    ('借阅规则是什么', 'policy_faq', 'faq', '借阅规则=5 vs 借阅=3，FAQ应胜出'),
    ('续借的规则是什么', 'borrow_manage', 'business',
     '续借=5（唯一命中），规则非关键词，走业务通道（正确行为，非缺陷）'),

    # === 场景2：业务关键词远多于 FAQ，业务应胜出 ===
    ('续借我的书', 'borrow_manage', 'business', '续借=5，纯业务意图'),
    ('续借缴费逾期', 'borrow_manage', 'business', '续借=5+逾期=4+欠费=4=13'),
    ('帮我还书', 'borrow_manage', 'business', '还书=5'),

    # === 场景3：纯 FAQ 意图 ===
    ('图书馆开放时间', 'policy_faq', 'faq', '开放时间=5'),
    ('怎么用知网', 'resource_guide', 'faq', '知网=5'),
    ('论文查重在哪里', 'resource_guide', 'faq', '论文查重=5'),
    ('服务台在哪', 'navigation', 'faq', '服务台在哪=5'),
    ('可以带咖啡进图书馆吗', 'policy_faq', 'faq', '能带=4+咖啡=4=8'),
    ('图书馆在寒假开放吗', 'policy_faq', 'faq', '寒假=3+开放=2=5'),
    ('图书馆暑假闭馆吗', 'policy_faq', 'faq', '暑假=3+闭馆=3=6'),
    ('查重怎么弄', 'resource_guide', 'faq', '查重=4'),

    # === 场景4：纯业务意图 ===
    ('三体有没有在架', 'book_search', 'business', '有没有在架=4'),
    ('找一本C++的书', 'book_search', 'business', '找一本=4，应命中'),
    ('想看三体', 'book_search', 'business', '想看=3'),
    ('预约明天的自习室', 'seat_reserve', 'business', '预约=3+自习室=4=7'),
    ('我要转人工', 'human_transfer', 'business', '人工=5'),
    ('预约一个座位', 'seat_reserve', 'business', '预约=3+座位=3=6'),

    # === 场景5：多意图混合 ===
    ('借阅和欠费的规则', 'borrow_manage', 'business', '借阅=3+欠费=4=7 vs 规则无命中'),

    # === 场景6：完全未知 → RAG ===
    ('今天天气怎么样', 'unknown', 'rag', '无任何关键词命中'),
    ('你能做什么', 'unknown', 'rag', '无任何关键词命中'),
    ('帮我写一首诗', 'unknown', 'rag', '无任何关键词命中'),

    # === 场景7：边界 case ===
    ('人工', 'human_transfer', 'business', '单关键词：人工=5'),
    ('投诉', 'human_transfer', 'business', '单关键词：投诉=5'),
    ('座位', 'seat_reserve', 'business', '单关键词：座位=3'),
]

print('=' * 90)
print(f'{"输入":<22} {"预期意图":<16} {"预期通道":<10} {"实际意图":<16} {"实际通道":<10} {"结果":<6}')
print('=' * 90)

passed = 0
failed = 0
for text, expect_intent, expect_channel, desc in test_cases:
    intent, score, channel = c.classify(text)
    ok = (intent == expect_intent) and (channel == expect_channel)
    status = 'PASS' if ok else 'FAIL'
    if ok:
        passed += 1
    else:
        failed += 1
    print(f'{text:<22} {expect_intent:<16} {expect_channel:<10} {intent:<16} {channel:<10} {status:<6} score={score}')
    if not ok:
        print(f'    >>> 期望 {expect_intent}/{expect_channel}，实际 {intent}/{channel}')
        print(f'    >>> 场景: {desc}')

print('=' * 90)
print(f'总计: {passed} passed, {failed} failed, {passed + failed} total')

# ========== 冲突消解机制详解 ==========
print('\n' + '=' * 90)
print('📊 冲突消解机制详解')
print('=' * 90)
print('规则: second_best_score >= best_score * 0.8')
print('  AND best是业务通道、second是FAQ通道 → 走FAQ')
print()

conflict_analysis = [
    ('借书规则是什么', 'borrow_manage', 3, 'policy_faq', 5, 'faq',
     'FAQ(5) >= business(3)*0.8=2.4 ✅ 触发FAQ降级'),
    ('借阅规则是什么', 'borrow_manage', 3, 'policy_faq', 5, 'faq',
     '同上，FAQ(5) >= 2.4 ✅'),
    ('续借借阅欠费', 'borrow_manage', 12, 'policy_faq', 0, 'business',
     'business独占，无冲突 ✅'),
    ('人工转接投诉客服', 'human_transfer', 14, 'policy_faq', 0, 'business',
     'human_transfer独占 ✅'),
    ('借书', 'borrow_manage', 0, 'policy_faq', 0, 'rag',
     '都没命中 → unknown → rag ✅'),
]

for text, i1, s1, i2, s2, exp_ch, desc in conflict_analysis:
    intent, score, channel = c.classify(text)
    match = '✅' if channel == exp_ch else '❌'
    print(f'  "{text}": business={s1}(borrow_manage) vs faq={s2}(policy_faq) → {channel} {match}')
    print(f'    {desc}')

# ========== 打分过程可视化 ==========
print('\n' + '=' * 90)
print('🔍 打分过程可视化')
print('=' * 90)

debug_cases = ['借书规则是什么', '三体有没有在架', '图书馆开放时间']
for text in debug_cases:
    print(f'\n【{text}】')
    text_lower = text.lower()
    from app.services.chat_service import INTENT_WEIGHTS
    for intent, rules in INTENT_WEIGHTS.items():
        total = 0
        hit_keywords = []
        for kw, weight in rules:
            if kw.lower() in text_lower:
                total += weight
                hit_keywords.append(f'{kw}(+{weight})')
        if total > 0:
            print(f'  {intent}: score={total}  命中: {", ".join(hit_keywords)}')
    result = c.classify(text)
    print(f'  → 最终: intent={result[0]}, score={result[1]}, channel={result[2]}')

print('\n' + '=' * 90)
if failed == 0:
    print('🎉 所有测试用例通过！')
else:
    print(f'⚠️  有 {failed} 个用例失败，请检查！')
