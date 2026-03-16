import os
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.corpora.dictionary import Dictionary
import jieba.posseg as pseg
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ====================== 核心配置（严格按论文参数） ======================
W2V_CONFIG = {
    'vector_size': 200,      # 词向量维度200
    'window': 5,             # 上下文窗口5
    'min_count': 2,          # 最小词频2
    'sg': 1,                 # 1=Skip-gram，0=CBOW
    'hs': 0,                 # 0=Negative Sampling，1=Hierarchical Softmax
    'negative': 5,           # 负采样数量
    'epochs': 10,            # 训练轮数10
    'workers': 4,            # 线程数
    'seed': 42               # 随机种子
}

# 城规领域自定义词典（核心术语）
CITY_PLAN_DICT = {
    '交通类': ['轨道交通', '公交站点', '路网密度', '通行效率', '拥堵指数'],
    '环境类': ['绿化率', '噪音污染', '空气质量', '生态廊道', '垃圾分类'],
    '城市更新': ['老旧小区改造', '容积率', '建筑密度', '配套设施', '海绵城市'],
    '隐喻映射': {
        '面子工程': '无实际功能的形象化建设项目',
        '最后一公里': '公共服务覆盖的末端短板问题',
        '踢皮球': '部门间责任推诿导致问题搁置',
        '大白象工程': '高成本低效益的冗余建设项目',
        '一刀切': '缺乏差异化的粗放式规划决策',
        '打补丁': '临时补救性的规划调整措施'
    }
}

class Word2VecMetaphorReplacer:
    def __init__(self):
        # 加载并添加城规领域词典
        self._load_city_plan_dict()
        # 初始化模型
        self.model = None
        self.word_vectors = None
        
    def _load_city_plan_dict(self):
        """加载城规领域自定义词典到jieba分词"""
        # 合并所有领域术语
        all_terms = []
        for category, terms in CITY_PLAN_DICT.items():
            if category != '隐喻映射':
                all_terms.extend(terms)
        # 添加自定义词典
        for term in all_terms:
            jieba.add_word(term)
        # 保存自定义词典文件（供Word2Vec使用）
        with open('city_plan_dict.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_terms))

    def train_model(self, corpus_path):
        """
        训练Word2Vec模型（Skip-gram+Negative Sampling）
        :param corpus_path: 城规评论文本语料路径（txt格式，每行一条评论）
        """
        print("开始训练Word2Vec模型...")
        # 加载语料并分词
        sentences = LineSentence(corpus_path)
        
        # 训练模型（严格按论文参数）
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=W2V_CONFIG['vector_size'],
            window=W2V_CONFIG['window'],
            min_count=W2V_CONFIG['min_count'],
            sg=W2V_CONFIG['sg'],
            hs=W2V_CONFIG['hs'],
            negative=W2V_CONFIG['negative'],
            epochs=W2V_CONFIG['epochs'],
            workers=W2V_CONFIG['workers'],
            seed=W2V_CONFIG['seed']
        )
        
        # 保存模型
        self.model.save('city_plan_word2vec.model')
        self.word_vectors = self.model.wv
        print("Word2Vec模型训练完成并保存至: city_plan_word2vec.model")
        
        # 验证领域术语语义对齐效果
        self._validate_domain_terms()
        
    def _validate_domain_terms(self):
        """验证城规领域术语的语义对齐效果"""
        print("\n=== 领域术语语义相似度验证 ===")
        test_terms = ['轨道交通', '绿化率', '老旧小区改造']
        for term in test_terms:
            if term in self.word_vectors:
                similar_terms = self.word_vectors.most_similar(term, topn=5)
                print(f"{term} 最相似的5个术语：")
                for sim_term, score in similar_terms:
                    print(f"  - {sim_term}: {score:.4f}")
                print()

    def load_bert_metaphor_result(self, metaphor_result_path):
        """
        加载BERT隐喻识别模块的输出结果
        :param metaphor_result_path: BERT输出的隐喻识别结果（json格式）
        :return: 隐喻识别结果列表
        """
        with open(metaphor_result_path, 'r', encoding='utf-8') as f:
            metaphor_results = json.load(f)
        return metaphor_results

    def _get_semantic_similar_word(self, word, topn=3):
        """
        获取与目标词语义最相似的城规领域术语
        :param word: 目标词
        :param topn: 返回最相似的n个词
        :return: 最相似的词列表
        """
        if word not in self.word_vectors:
            return []
        return self.word_vectors.most_similar(word, topn=topn)

    def replace_metaphor(self, text, metaphor_words):
        """
        隐喻替换核心函数：保留原语义，将隐喻词替换为城规领域规范表达
        :param text: 输入文本（BERT处理后的规范文本）
        :param metaphor_words: BERT识别出的隐喻词列表
        :return: 替换后的文本、替换详情
        """
        replaced_text = text
        replace_details = []
        
        # 1. 优先使用固定隐喻映射表替换
        metaphor_map = CITY_PLAN_DICT['隐喻映射']
        for metaphor in metaphor_words:
            if metaphor in metaphor_map:
                # 固定映射替换
                standard_expr = metaphor_map[metaphor]
                replaced_text = replaced_text.replace(metaphor, standard_expr)
                replace_details.append({
                    'metaphor': metaphor,
                    'replaced_with': standard_expr,
                    'type': '固定映射'
                })
            else:
                # 2. 基于Word2Vec语义相似度替换
                similar_terms = self._get_semantic_similar_word(metaphor, topn=1)
                if similar_terms:
                    standard_expr = similar_terms[0][0]
                    replaced_text = replaced_text.replace(metaphor, standard_expr)
                    replace_details.append({
                        'metaphor': metaphor,
                        'replaced_with': standard_expr,
                        'similarity': similar_terms[0][1],
                        'type': '语义相似度'
                    })
        
        return replaced_text, replace_details

    def batch_process(self, metaphor_result_path):
        """
        批量处理隐喻替换
        :param metaphor_result_path: BERT隐喻识别结果路径
        :return: 替换结果列表
        """
        print("\n开始批量处理隐喻替换...")
        # 加载BERT隐喻识别结果
        metaphor_results = self.load_bert_metaphor_result(metaphor_result_path)
        
        final_results = []
        for item in tqdm(metaphor_results, desc="处理进度"):
            original_text = item['original_text']
            metaphor_words = item['metaphor_words']
            
            # 执行隐喻替换
            replaced_text, replace_details = self.replace_metaphor(original_text, metaphor_words)
            
            # 保存结果
            final_results.append({
                'original_text': original_text,
                'metaphor_words': metaphor_words,
                'replaced_text': replaced_text,
                'replace_details': replace_details
            })
        
        # 保存替换结果
        with open('metaphor_replaced_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        
        # 输出替换效果示例
        self._show_replace_examples(final_results[:5])
        
        return final_results

    def _show_replace_examples(self, results):
        """展示替换效果示例"""
        print("\n=== 隐喻替换效果示例 ===")
        for i, result in enumerate(results):
            print(f"\n示例 {i+1}:")
            print(f"原始文本：{result['original_text']}")
            print(f"识别的隐喻词：{result['metaphor_words']}")
            print(f"替换后文本：{result['replaced_text']}")
            if result['replace_details']:
                print("替换详情：")
                for detail in result['replace_details']:
                    if detail['type'] == '固定映射':
                        print(f"  - {detail['metaphor']} → {detail['replaced_with']}（固定映射）")
                    else:
                        print(f"  - {detail['metaphor']} → {detail['replaced_with']}（相似度：{detail['similarity']:.4f}）")

# ====================== 测试与演示 ======================
def create_demo_corpus():
    """创建演示用城规评论文本语料"""
    demo_corpus = [
        "这个面子工程完全不考虑实际的路网密度和通行效率",
        "最后一公里的问题导致公交站点覆盖不足",
        "部门之间互相踢皮球，绿化率提升计划迟迟不落地",
        "这个大白象工程浪费了大量资金，不如投入老旧小区改造",
        "一刀切的规划方式忽略了不同区域的生态廊道建设需求",
        "轨道交通规划存在打补丁的情况，缺乏系统性",
        "路网密度低导致早晚高峰拥堵指数居高不下",
        "海绵城市建设能有效提升城市的雨水调蓄能力",
        "建筑密度过高影响了居民的生活质量和空气质量",
        "垃圾分类设施配套不足是当前城市环境治理的短板"
    ]
    # 保存演示语料
    with open('city_plan_corpus.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(demo_corpus))
    
    # 创建演示用BERT隐喻识别结果
    demo_bert_results = [
        {
            "original_text": "这个面子工程完全不考虑实际的路网密度和通行效率",
            "metaphor_words": ["面子工程"]
        },
        {
            "original_text": "最后一公里的问题导致公交站点覆盖不足",
            "metaphor_words": ["最后一公里"]
        },
        {
            "original_text": "部门之间互相踢皮球，绿化率提升计划迟迟不落地",
            "metaphor_words": ["踢皮球"]
        }
    ]
    with open('bert_metaphor_results.json', 'w', encoding='utf-8') as f:
        json.dump(demo_bert_results, f, ensure_ascii=False, indent=4)
    
    return 'city_plan_corpus.txt', 'bert_metaphor_results.json'

if __name__ == "__main__":
    # 1. 创建演示语料和BERT隐喻识别结果
    corpus_path, metaphor_result_path = create_demo_corpus()
    
    # 2. 初始化Word2Vec隐喻替换器
    replacer = Word2VecMetaphorReplacer()
    
    # 3. 训练Word2Vec模型
    replacer.train_model(corpus_path)
    
    # 4. 批量处理隐喻替换
    final_results = replacer.batch_process(metaphor_result_path)
    
    # 5. 保存最终规范化语料
    with open('normalized_city_plan_corpus.txt', 'w', encoding='utf-8') as f:
        for result in final_results:
            f.write(result['replaced_text'] + '\n')
    print("\n规范化城规评论文料已保存至: normalized_city_plan_corpus.txt")