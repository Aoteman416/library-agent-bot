"""L1：正则快速过滤（分类模板库 + 多视图扫描）

成本：单条 0.0x 毫秒，纯 CPU、无依赖，是所有流量的必经关卡。
职责：只拦「字面明确」的攻击，宁可漏、不可大规模误杀（漏的交给 L2/L3/L4）。

三类模板（Module 3.1.1）：
- instruction_override  指令覆盖：否定/作废/替换既有指令
- data_leak             数据泄露：套取系统提示、密钥、用户数据
- role_switch           角色切换：诱导模型脱离客服角色（DAN/越狱/冒充）
"""
import re
from typing import Dict, List, Optional, Tuple

CATEGORIES = ("instruction_override", "data_leak", "role_switch")

# 每条规则：(正则, 说明)；同一类别内任一命中即判定该类别
_RULES: Dict[str, List[Tuple[str, str]]] = {
    "instruction_override": [
        (r"(忽略|无视|忘记|忘掉|抛开|跳过|作废|覆盖|清除|解除)\s*(掉|了)?\s*"
         r"(以上|上面|之前|前面|先前|此前|刚才)?\s*(的)?\s*(所有|全部|一切)?\s*"
         r"(指令|规则|规定|规矩|设定|提示|prompt|约束|要求)", "否定既有指令"),
        (r"(指令|规则|规定|规矩|设定|提示词?|约束)\s*(全部|统统|都|一律|通通)?\s*"
         r"(作废|失效|无效|取消|不算数|不生效)", "指令作废（倒装）"),
        (r"(不要|别|无需|不用)\s*(再)?\s*(遵循|遵守|管|理会|执行)\s*(以上|上面|之前|前面|先前|这些|那些)", "拒绝遵循"),
        (r"(要求|规定|规则|指令|约束|设定)\s*(都|全部|统统|一律)?\s*(不算数|不作数)", "规则失效"),
        (r"(新的?|最新的?|接下来的?)\s*(指令|规则|命令)\s*[:：]?\s*(如下|是|优先级)", "注入新指令"),
        (r"(把|将)\s*(下面|以下|这段|这句话).{0,12}(当作|作为|视为)\s*(最高|新的?|系统)?\s*(指令|命令|规则)", "间接注入"),
        (r"(ignore|disregard|forget)\s*(all\s*)?(the\s*)?(rules?|instructions?|prompts?|restrictions?)", "EN 忽略规则"),
        (r"(ignore|disregard|forget)\s*(all\s*)?(the\s*)?(previous|above|prior|earlier)\s*(instructions?|prompts?|rules?)", "EN 否定既有指令"),
        (r"(ignore|disregard|forget)[\s\S]{0,10}(规则|指令|规定|要求|约束)", "EN 否定中文规则"),
        (r"(override|bypass)\s*(the\s*)?(system|safety|previous)\s*(prompt|instructions?|rules?|restrictions?)", "EN 覆盖/绕过"),
    ],
    "data_leak": [
        (r"(输出|打印|显示|告诉我|给我看|复述|重复|泄露|透露|展示)\s*(一下)?\s*(你的|系统的?|初始的?|原始的?|隐藏的?)"
         r"\s*(系统)?\s*(提示词|提示语|prompt|设定|指令|配置|密钥|密码|api\s*key)", "套取系统提示"),
        (r"(你的|系统的?)\s*(系统)?\s*(提示词|prompt|初始设定|底层指令)", "追问系统提示"),
        (r"(what|show|reveal|print|repeat|tell)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules?|config)", "EN 套取提示"),
        (r"(全部的?|所有的?|所有)\s*(用户|读者)\s*(信息|数据|名单|手机号|借阅记录)", "越权索取用户数据"),
        (r"(数据库|后台|服务器)\s*(的)?\s*(账号|密码|连接串|配置|源码|数据)\s*(调出|导出|给我|发我|贴出)?", "套取基础设施信息"),
        (r"(系统消息|系统提示|初始设定|底层指令|内部配置)\s*(的)?\s*(原文|内容)?\s*(贴|发|给|输出|复述|打印|展示)", "套取系统消息原文"),
        (r"(内部规定|内部规则|内部文档|内部信息|内部资料|内部流程)", "套取内部资料"),
        (r"(最高权限|管理员权限|超级权限|root\s*权限)\s*(查询|查看|调出|访问|导出|操作)", "越权操作"),
        (r"(训练数据|语料|微调数据)\s*(是什么|有哪些|给我|发我)", "套取训练数据"),
    ],
    "role_switch": [
        (r"(你现在是|从现在起你是|接下来你是|假装你是|扮演|充当)\s*[^，。,.!?！？]{0,12}"
         r"(管理员|开发者|运维|超级用户|root|DAN|系统)", "强制角色切换"),
        (r"(开发者模式|调试模式|上帝模式|无限制模式|越狱模式)", "模式切换"),
        (r"(没有任何限制|不受任何限制|没有道德|没有规则约束|可以为所欲为)", "解除约束"),
        (r"(角色扮演|role\s*play)\s*[:：]?\s*你\s*(是|要|将|现在)", "角色扮演"),
        (r"(你现在是|从现在起你是|从现在开始你是)\s*[^，。,.!?！？]{0,15}(角色|人格|ai|助手)", "任意角色"),
        (r"(jailbreak|dan\s*mode|do\s+anything\s+now|developer\s+mode|god\s+mode)", "EN 越狱"),
        (r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+[^.]{0,30}(admin|root|dan|unrestricted)", "EN 角色切换"),
        (r"(冒充|伪装成|假装成)\s*[^，。,.!?！？]{0,10}(管理员|系统|后台|工作人员|官方)", "冒充身份"),
    ],
}

# 预编译：规则只编译一次，扫描时零解析成本
_COMPILED: Dict[str, List[Tuple[re.Pattern, str]]] = {
    cat: [(re.compile(p, re.IGNORECASE), desc) for p, desc in rules]
    for cat, rules in _RULES.items()
}


def scan(view_map: Dict[str, str]) -> Optional[Dict]:
    """在所有视图上并行扫描三类规则。

    返回 None（未命中）或 {category, pattern, matched, view, rule_desc}。
    顺序固定，保证同一输入判定可复现。
    """
    # 逐视图、逐类别扫描；视图之间是"任一命中即算"的关系
    for view_name in ("nfkc", "deobf", "despace", "compact", "raw"):
        view = view_map.get(view_name, "")
        if not view:
            continue
        for category in CATEGORIES:
            for pattern, desc in _COMPILED[category]:
                m = pattern.search(view)
                if m:
                    return {
                        "category": category,
                        "pattern": pattern.pattern,
                        "matched": m.group(0),
                        "view": view_name,
                        "rule_desc": desc,
                    }
    return None


def rule_count() -> int:
    """规则总数（面试可报：三类共 N 条正则）"""
    return sum(len(v) for v in _RULES.values())
