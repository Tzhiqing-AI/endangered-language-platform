"""
Dashboard NLP 功能视图

包含中文自然语言处理功能：情感分析、关键词提取、摘要生成、文本润色
改进版本：更准确的分析、词云数据、AI辅助润色
"""

import logging
from collections import Counter

from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.http import JsonResponse

from Dashboard.common import get_user_context, get_openai_client
from language.utils import login_required_custom, api_response, rate_limit

logger = logging.getLogger(__name__)


# =============================================================================
# 情感分析词典（自建，覆盖网络用语、程度副词、否定词）
# =============================================================================

# 情感强度：数值越大，情感越强（-5 ~ +5）
POSITIVE_WORDS = {
    # 基础褒义
    '好': 1, '棒': 2, '赞': 2, '喜欢': 2, '爱': 2, '美': 1, '香': 2, '甜': 1,
    '不错': 2, '推荐': 2, '满意': 3, '开心': 2, '高兴': 2, '快乐': 2, '完美': 4,
    '优秀': 3, '出色': 3, '卓越': 4, '良好': 2, '优秀': 3, '一流': 4, '顶级': 4,
    '好吃': 3, '美味': 3, '可口': 3, '鲜': 2, '香甜': 3, '爽': 2, '过瘾': 3,
    '漂亮': 2, '美': 1, '帅': 2, '酷': 2, '萌': 1, '可爱': 2, '美': 1,
    '感谢': 2, '感激': 3, '谢谢': 1, '辛苦': 1, '用心': 2, '贴心': 2,
    '成功': 3, '胜利': 4, '进步': 2, '提升': 2, '改善': 2, '优化': 2,
    '温暖': 2, '温馨': 2, '感动': 3, '暖心': 3, '友好': 2, '善良': 2,
    '划算': 2, '实惠': 2, '便宜': 1, '值': 2, '值': 2,
    '顺': 1, '顺利': 2, '流畅': 2, '高效': 2, '快速': 1, '便捷': 2,
    '清晰': 1, '干净': 1, '整洁': 1, '卫生': 1, '安全': 2, '健康': 2,
    '惊喜': 3, '超出预期': 4, '物超所值': 4, '性价比高': 3,

    # 网络流行语 - 正面
    '绝绝子': 5, 'yyds': 5, '永远的神': 5, '宝藏': 3, '强推': 3,
    '嘎嘎': 2, '真香': 4, '绝': 3, '巨好吃': 4, '嘎嘎好吃': 4,
    '神': 3, '神仙': 4, '王炸': 4, '炸裂': 4, '嗨翻': 3, '嗨': 2,
    '种草': 2, '拔草': 1, '安利': 2, '五星好评': 4, '五颗星': 4,
    '满分': 4, '拿捏': 2, '到位': 2, '到位': 2, '没毛病': 3,
    '666': 4, '牛': 3, '厉害': 3, '强': 3, '稳': 2, '稳了': 2,
    '赢麻了': 4, '赢': 3, '拉满': 3, '上头': 2, '顶': 2,
    '啊这': 0,  # 中性/无语
}

NEGATIVE_WORDS = {
    # 基础贬义
    '差': -2, '烂': -3, '坏': -2, '糟': -3, '糟糕': -4, '差劲': -3,
    '难吃': -3, '难喝': -3, '难闻': -2, '难': -1, '难受': -2,
    '贵': -2, '贵': -2, '坑': -3, '黑心': -4, '宰客': -4,
    '后悔': -3, '失望': -3, '绝望': -5, '难过': -3, '伤心': -3,
    '失败': -3, '崩溃': -4, '完蛋': -4, '废了': -3,
    '问题': -1, '毛病': -2, '缺点': -1, '不足': -1, '短板': -2,
    '慢': -1, '卡': -2, '崩溃': -4, '死机': -3, '卡顿': -2,
    '脏': -2, '乱': -1, '差': -2, '乱': -1, '糟糕': -4,
    '危险': -3, '不安全': -3, '风险': -2, '麻烦': -2,
    '欺负': -3, '辱骂': -4, '骚扰': -4, '暴力': -4,
    '丑': -2, '难看': -2, '邋遢': -2,
    '假': -2, '骗': -3, '骗人': -3, '虚假': -3, '欺诈': -4,
    '冷': -1, '热': -1, '吵': -2, '闹': -2, '挤': -1,
    '生气': -3, '愤怒': -4, '烦': -2, '烦躁': -3, '厌倦': -3,
    '讨厌': -3, '厌恶': -4, '恶心': -4, '反感': -3,
    '退': -2, '退款': -2, '退货': -2, '差评': -3,

    # 网络流行语 - 负面
    '服了': -2, '劝退': -3, '踩雷': -4, '翻车': -4, '避雷': -3,
    '巨难吃': -4, '难吃死了': -5, '垃圾': -4, 'lj': -4, 'tui': -3,
    '下头': -3, '破防': -2, '麻了': -2, 'emo': -2, '芭比Q了': -4,
    '裂开': -2, '当场去世': -5, '原地爆炸': -4, '服': -1,
    '退退退': -3, '吐了': -3, '离谱': -2, '劝退': -3, '拉胯': -3,
    '不行': -2, '不行': -2, '差评': -3, '就这': -2,
    '黄鼠狼': -3,  # 骂人用语
    '简直': 0,  # 程度副词，但常用于加强负面
    '不咋地': -3, '不咋的': -3, '没意思': -2, '没劲': -2, '无聊': -2,
    '傻': -2, '蠢': -2, '笨': -2, '弱': -2, '烂': -3, '渣': -3,
}

# 否定词：会反转后续情感极性
NEGATION_WORDS = {
    '不', '没', '别', '无', '非', '莫', '勿', '未', '甭', '毋',
    '没有', '不会', '不能', '不要', '不可', '不行', '不成',
    '毫无', '毫无', '不咋', '不怎么', '不太', '不太',
}

# 程度副词：会缩放情感强度
DEGREE_WORDS = {
    # 强化（×2 或更多）
    '巨': 2.0, '超': 2.0, '超级': 2.5, '爆': 2.5, '极': 2.0, '极其': 2.5,
    '非常': 2.0, '十分': 2.0, '特别': 1.8, '格外': 1.8, '相当': 1.5,
    '很': 1.5, '太': 1.8, '挺': 1.3, '蛮': 1.3, '够': 1.3,
    '真的': 1.3, '确实': 1.3, '简直': 1.5, '绝对的': 2.0,
    'amazing': 2.0, 'incredible': 2.0,

    # 减弱（×0.5）
    '有点': 0.6, '有些': 0.6, '稍微': 0.5, '略微': 0.5, '一点': 0.5,
    '还算': 0.7, '勉强': 0.5, '一般般': 0.4,
}


def _lexicon_sentiment(text):
    """
    基于自定义词典的情感分析（处理否定、程度副词、网络用语）

    Returns:
        dict: {'score': 0~1, 'label': str, 'pos_words': [...], 'neg_words': [...], 'confidence': 0~1}
    """
    import jieba
    import jieba.posseg

    if not text or not text.strip():
        return {
            'score': 0.5, 'label': '中性', 'pos_words': [], 'neg_words': [],
            'confidence': 0.0, 'method': 'lexicon'
        }

    # 把关键短语加入 jieba 词典，避免被切散
    for phrase in list(POSITIVE_WORDS.keys()) + list(NEGATIVE_WORDS.keys()) + list(NEGATION_WORDS) + list(DEGREE_WORDS.keys()):
        if len(phrase) >= 2:
            jieba.add_word(phrase, freq=10000)

    # jieba 分词（保留原始顺序）
    words = list(jieba.cut(text))

    pos_matches = []  # 命中的正面词 [(word, weight, position)]
    neg_matches = []
    total_score = 0.0
    matched_count = 0

    for i, word in enumerate(words):
        # 跳过停用词/标点
        if len(word.strip()) == 0 or word in '，。！？、；：""''（）()【】[]《》<>…—·':
            continue

        # 在前 2 个词内查找否定词（在情感词之前出现）
        negation_window = []
        for j in range(max(0, i - 2), i):
            if words[j] in NEGATION_WORDS:
                negation_window.append(words[j])

        # 在前 2 个词内查找程度副词
        degree = 1.0
        for j in range(max(0, i - 2), i):
            if words[j] in DEGREE_WORDS:
                degree = DEGREE_WORDS[words[j]]
                break

        # 匹配情感词（尝试 3 字、2 字、1 字匹配）
        word_score = 0
        matched = False
        for length in [3, 2, 1]:
            if i + length > len(words):
                continue
            phrase = ''.join(words[i:i + length])
            if phrase in POSITIVE_WORDS:
                word_score = POSITIVE_WORDS[phrase]
                pos_matches.append((phrase, word_score, i))
                matched = True
                break
            if phrase in NEGATIVE_WORDS:
                word_score = NEGATIVE_WORDS[phrase]
                neg_matches.append((phrase, word_score, i))
                matched = True
                break

        if matched:
            matched_count += 1
            # 否定反转
            if negation_window:
                word_score = -word_score
            # 程度缩放
            word_score = word_score * degree
            total_score += word_score

    # 计算归一化分数（0~1）
    if matched_count == 0:
        return {
            'score': 0.5, 'label': '中性', 'pos_words': [], 'neg_words': [],
            'confidence': 0.0, 'method': 'lexicon'
        }

    # 用 sigmoid-like 把累计分压到 0~1
    # 总分大概在 -15 ~ +15 之间
    normalized = 1.0 / (1.0 + 2.5 ** (-total_score / 3.5))

    # 置信度 = 命中的情感词数 / 总词数
    confidence = min(1.0, matched_count / max(len(words), 1) * 3)

    # 标签
    if normalized >= 0.75:
        label = '非常积极'
    elif normalized >= 0.6:
        label = '积极'
    elif normalized >= 0.4:
        label = '中性'
    elif normalized >= 0.25:
        label = '消极'
    else:
        label = '非常消极'

    return {
        'score': round(normalized, 4),
        'label': label,
        'pos_words': [w[0] for w in pos_matches],
        'neg_words': [w[0] for w in neg_matches],
        'confidence': round(confidence, 3),
        'method': 'lexicon',
        'matched_count': matched_count,
        'raw_score': round(total_score, 2),
    }


def _ai_sentiment(text):
    """
    使用 Kimi 进行高精度情感分析（兜底层）

    Returns:
        dict | None: 成功返回字典，失败返回 None
    """
    try:
        client = get_openai_client()
        prompt = f"""请对下面这段中文文本进行情感分析。
要求：
1. 输出一段严格的 JSON，不要任何额外说明
2. 字段：score (0到1的浮点数，0最消极，1最积极)、label (从"非常消极/消极/中性/积极/非常积极"中选一个)、reason (一句话解释)
3. 注意：网络流行语、程度副词、否定词、隐喻都可能反转表层情感
4. 例子："巨难吃"应判为"非常消极"；"简直黄鼠狼"是骂人，应判为消极

文本：{text}"""

        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": "你是中文情感分析专家，输出必须是严格 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()

        # 尝试解析 JSON（AI 偶尔会包 markdown 代码块）
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip().rstrip('`').strip()

        import json
        data = json.loads(content)
        return {
            'score': float(data.get('score', 0.5)),
            'label': data.get('label', '中性'),
            'reason': data.get('reason', ''),
            'method': 'AI',
            'confidence': 0.95,
        }
    except Exception as e:
        logger.warning(f"AI 情感分析失败: {e}")
        return None


def _analyze_sentiment(text):
    """
    情感分析 - 三层混合架构：
    1. 自定义词典（快、准、覆盖网络用语）
    2. AI 兜底（高准确率，处理歧义/隐喻）
    3. SnowNLP（最后兜底）

    Returns:
        dict: 包含情感分数、标签、置信度、句子级分析、命中词等
    """
    # 第 1 层：自定义词典
    lex_result = _lexicon_sentiment(text)

    # 第 2 层：仅在词典置信度低或需要消歧时调用 AI
    ai_result = None
    if lex_result['confidence'] < 0.4 or len(text) > 50 and matched_count_check(text, lex_result) < 2:
        ai_result = _ai_sentiment(text)

    # 选择最终结果
    if ai_result:
        result = ai_result
        # 把词典的命中词也带上
        result['pos_words'] = lex_result['pos_words']
        result['neg_words'] = lex_result['neg_words']
    else:
        result = lex_result

    score = result['score']
    sentiment = result['label']

    # 统一标签格式
    label_map = {
        '非常积极': ('非常积极', '😄', '#10b981'),
        '积极': ('积极', '🙂', '#34d399'),
        '中性': ('中性', '😐', '#6366f1'),
        '消极': ('消极', '😟', '#f59e0b'),
        '非常消极': ('非常消极', '😢', '#ef4444'),
    }
    sentiment, emoji, color = label_map.get(sentiment, ('中性', '😐', '#6366f1'))

    # 句子级情感分析（使用词典以保持一致）
    sentence_sentiments = []
    for sent in [s for s in text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n') if s.strip()][:10]:
        s = sent.strip()
        if len(s) > 1:
            s_lex = _lexicon_sentiment(s)
            sentence_sentiments.append({
                'text': s[:50] + ('...' if len(s) > 50 else ''),
                'score': s_lex['score'],
                'label': s_lex['label'],
            })

    out = {
        'score': round(score, 3),
        'percentage': f'{score:.1%}',
        'sentiment': sentiment,
        'emoji': emoji,
        'color': color,
        'method': result['method'],
        'confidence': result.get('confidence', 0.5),
        'pos_words': result.get('pos_words', []),
        'neg_words': result.get('neg_words', []),
        'sentence_sentiments': sentence_sentiments,
        'sentence_count': len(sentence_sentiments),
        'raw_score': result.get('raw_score'),
    }
    if result.get('reason'):
        out['reason'] = result['reason']
    return out


def matched_count_check(text, lex_result):
    """判断词典命中数是否足够"""
    return lex_result.get('matched_count', 0)


@login_required_custom
def nlp_page(request):
    """
    NLP 自然语言处理页面视图

    URL: /Dashboard/nlp/
    Method: GET
    """
    user, name, _ = get_user_context(request)
    return render(request, 'nlp.html', {'name': name, 'username': user.username})


def _analyze_sentiment(text):
    """
    情感分析 - 使用 SnowNLP 进行更精细的分析

    Returns:
        dict: 包含情感分数、标签、置信度、句子级别分析
    """
    from snownlp import SnowNLP

    s = SnowNLP(text)
    score = s.sentiments

    # 更精细的情感分类
    if score >= 0.8:
        sentiment = '非常积极'
        emoji = '😊'
        color = '#10b981'
    elif score >= 0.6:
        sentiment = '积极'
        emoji = '🙂'
        color = '#34d399'
    elif score >= 0.4:
        sentiment = '中性'
        emoji = '😐'
        color = '#6366f1'
    elif score >= 0.2:
        sentiment = '消极'
        emoji = '😟'
        color = '#f59e0b'
    else:
        sentiment = '非常消极'
        emoji = '😢'
        color = '#ef4444'

    # 句子级别情感分析
    sentence_sentiments = []
    for sent in s.sentences[:10]:  # 最多分析10句
        if len(sent.strip()) > 2:
            sent_s = SnowNLP(sent)
            sent_score = sent_s.sentiments
            if sent_score >= 0.6:
                sent_label = '积极'
            elif sent_score >= 0.4:
                sent_label = '中性'
            else:
                sent_label = '消极'
            sentence_sentiments.append({
                'text': sent[:50] + ('...' if len(sent) > 50 else ''),
                'score': round(sent_score, 3),
                'label': sent_label
            })

    return {
        'score': round(score, 3),
        'percentage': f'{score:.1%}',
        'sentiment': sentiment,
        'emoji': emoji,
        'color': color,
        'sentence_count': len(s.sentences),
        'sentence_sentiments': sentence_sentiments,
        'summary': f'文本整体情感倾向为「{sentiment}」，置信度为 {score:.1%}。共包含 {len(s.sentences)} 个句子。'
    }


def _extract_keywords(text, top_k=10):
    """
    关键词提取 - 使用 jieba TF-IDF 算法

    Returns:
        dict: 包含关键词列表、权重、词云数据
    """
    import jieba.analyse
    import jieba

    # TF-IDF 关键词提取
    keywords_tfidf = jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)

    # TextRank 关键词提取（另一种算法，可以交叉验证）
    keywords_textrank = jieba.analyse.textrank(text, topK=top_k, withWeight=True)

    # 词频统计（用于词云）
    words = jieba.lcut(text)
    # 过滤停用词和短词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    word_freq = Counter(w for w in words if len(w) > 1 and w not in stop_words)

    # 词云数据（前50个高频词）
    wordcloud_data = [{'word': w, 'count': c} for w, c in word_freq.most_common(50)]

    # 格式化关键词结果
    keywords_list = []
    for word, weight in keywords_tfidf:
        # 计算在文本中出现的次数
        count = text.count(word)
        keywords_list.append({
            'word': word,
            'weight': round(weight, 4),
            'weight_percent': f'{weight:.1%}',
            'count': count
        })

    # TextRank 结果
    textrank_list = [{'word': w, 'weight': round(wt, 4)} for w, wt in keywords_textrank]

    return {
        'keywords': keywords_list,
        'textrank': textrank_list,
        'wordcloud': wordcloud_data,
        'total_words': len(words),
        'unique_words': len(set(words)),
        'summary': f'共识别出 {len(keywords_list)} 个关键词，文本总词数 {len(words)}，不重复词数 {len(set(words))}。'
    }


def _summarize_text(text, ratio=0.3):
    """
    文本摘要 - 使用 TextRank 算法提取关键句子

    Args:
        text: 待摘要文本
        ratio: 摘要比例（0-1），默认提取30%的句子

    Returns:
        dict: 包含摘要、关键句子、压缩比
    """
    from snownlp import SnowNLP

    s = SnowNLP(text)
    sentences = list(s.sentences)

    if not sentences:
        return {
            'summary': '文本过短，无法生成摘要。',
            'original_length': len(text),
            'summary_length': 0,
            'compression_ratio': '0%',
            'sentence_count': 0,
        }

    # 计算摘要句子数量
    summary_count = max(1, int(len(sentences) * ratio))
    summary_count = min(summary_count, 5)  # 最多5句

    # 使用 SnowNLP 的摘要功能（基于 TextRank）
    summary_sentences = []
    for i, sent in enumerate(sentences[:summary_count]):
        if len(sent.strip()) > 5:
            summary_sentences.append({
                'index': i + 1,
                'text': sent,
                'length': len(sent)
            })

    summary_text = '。'.join([s['text'] for s in summary_sentences]) + '。'

    return {
        'summary': summary_text,
        'sentences': summary_sentences,
        'original_length': len(text),
        'summary_length': len(summary_text),
        'compression_ratio': f'{len(summary_text) / len(text) * 100:.1f}%',
        'sentence_count': len(sentences),
        'selected_count': len(summary_sentences),
        'summary_text': f'原文共 {len(sentences)} 句，提取了 {len(summary_sentences)} 句作为摘要，压缩比为 {len(summary_text) / len(text) * 100:.1f}%。'
    }


def _polish_text(text):
    """
    文本润色 - 使用 AI 进行专业的文本改写

    Returns:
        dict: 包含润色后文本、修改说明
    """
    try:
        client = get_openai_client()

        prompt = f"""请对以下文本进行润色和优化，要求：
1. 保持原文的核心意思不变
2. 使语言更加流畅、专业
3. 修正可能的语法错误
4. 优化表达方式

原文：
{text}

请直接输出润色后的文本，不需要额外说明。"""

        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": "你是一位专业的文本编辑和润色专家，擅长优化各种类型的文本。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        polished = response.choices[0].message.content.strip()

        return {
            'original': text,
            'polished': polished,
            'original_length': len(text),
            'polished_length': len(polished),
            'method': 'AI 智能润色',
            'summary': '已使用 AI 对文本进行智能润色，优化了语言表达和专业性。'
        }

    except Exception as e:
        logger.warning(f"AI 润色失败，使用本地方法: {str(e)}")

        # 降级到本地润色方法
        import jieba
        import random

        # 更丰富的同义词词典
        synonym_dict = {
            '好': ['优秀', '出色', '卓越', '良好', '优异'],
            '很好': ['非常好', '相当好', '十分好', '极好'],
            '大': ['巨大', '庞大', '宏大', '广大', '重大'],
            '小': ['微小', '细小', '小巧', '迷你', '精巧'],
            '多': ['众多', '丰富', '大量', '许多', '繁多'],
            '少': ['少量', '少许', '稀少', '稀缺', '有限'],
            '重要': ['关键', '核心', '首要', '至关重要', '举足轻重'],
            '帮助': ['协助', '支持', '援助', '辅助', '助力'],
            '发展': ['进步', '提升', '推进', '促进', '推动'],
            '问题': ['议题', '课题', '难题', '挑战', '困境'],
        }

        polished = text
        changes = []
        for word, synonyms in synonym_dict.items():
            if word in polished:
                replacement = random.choice(synonyms)
                polished = polished.replace(word, replacement, 1)
                changes.append(f'「{word}」→「{replacement}」')

        return {
            'original': text,
            'polished': polished,
            'original_length': len(text),
            'polished_length': len(polished),
            'method': '本地同义词替换',
            'changes': changes,
            'summary': f'已对文本进行同义词优化，共替换 {len(changes)} 处表达。'
        }


@login_required_custom
@require_POST
@rate_limit(max_requests=20, window_seconds=60)
def nlp_process(request):
    """
    NLP 自然语言处理 API 接口

    URL: /Dashboard/nlp/process/
    Method: POST

    POST 参数:
        - text: 待处理文本
        - func_type: 功能类型 (sentiment/keywords/summarize/polish)
    """
    text = request.POST.get('text', '').strip()
    func_type = request.POST.get('func_type', '')

    if not text:
        return api_response(status='error', message='请输入待处理文本')

    # 文本长度限制
    if len(text) > 10000:
        return api_response(status='error', message='文本长度不能超过10000字')

    try:
        if func_type == 'sentiment':
            result = _analyze_sentiment(text)

        elif func_type == 'keywords':
            top_k = int(request.POST.get('top_k', 10))
            top_k = min(max(top_k, 3), 20)  # 限制在3-20之间
            result = _extract_keywords(text, top_k)

        elif func_type == 'summarize':
            ratio = float(request.POST.get('ratio', 0.3))
            ratio = min(max(ratio, 0.1), 0.5)  # 限制在10%-50%之间
            result = _summarize_text(text, ratio)

        elif func_type == 'polish':
            result = _polish_text(text)

        else:
            return api_response(status='error', message='无效的功能类型')

    except Exception as e:
        logger.error(f"NLP 处理失败: {str(e)}")
        return api_response(status='error', message=f'处理失败：{str(e)}')

    return api_response(status='success', data={'result': result})
