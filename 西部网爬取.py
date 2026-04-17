import requests
import csv
import time
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

# ================== 配置信息 ==================
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'http://news.cnwest.com/',
}
REQUEST_DELAY = 1  # 请求间隔（秒），避免对服务器造成压力

# ================== 新闻列表页解析 ==================
def get_article_urls_from_list(list_url: str, max_articles: int = 50) -> List[str]:
    urls = set()
    try:
        resp = requests.get(list_url, headers=HEADERS, timeout=15)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 匹配形如 /sxxw/a/2026/04/17/数字.html 的链接
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if re.search(r'/sxxw/a/\d{4}/\d{2}/\d{2}/\d+\.html', href):
                if href.startswith('http'):
                    full_url = href
                else:
                    full_url = f"http://news.cnwest.com{href}"
                urls.add(full_url)
                if len(urls) >= max_articles:
                    break
        print(f"从列表页 {list_url} 提取到 {len(urls)} 篇文章")
    except Exception as e:
        print(f"获取列表页失败 {list_url}: {e}")
    return list(urls)


# ================== 新闻正文提取 ==================
def extract_news_text(article_url: str) -> Optional[Dict]:
    """提取单篇新闻的标题、正文、发布时间等元信息。"""
    try:
        resp = requests.get(article_url, headers=HEADERS, timeout=15)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 标题
        title_tag = soup.find('title')
        title = title_tag.get_text().replace(' - 西部网（陕西新闻网）', '') if title_tag else ''

        # 正文内容（西部网正文通常在 div.content 或 .article-content 等容器内）
        content_div = soup.find('div', class_='content') or soup.find('div', class_='article-content')
        if not content_div:
            # 降级：提取所有文本段落
            paragraphs = soup.find_all('p')
            text = '\n'.join(p.get_text(strip=True) for p in paragraphs)
        else:
            text = content_div.get_text(strip=True)

        # 简单清洗：去掉多余空白、过短段落
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 50:   
            return None

        # 提取发布时间（西部网常见格式：2026-04-17 08:21）
        time_meta = soup.find('meta', {'name': 'publishdate'}) or soup.find('meta', {'property': 'article:published_time'})
        pub_time = ''
        if time_meta and time_meta.get('content'):
            pub_time = time_meta['content']
        else:
            # 降级：从页面文本中找日期
            date_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}'
            match = re.search(date_pattern, soup.get_text())
            if match:
                pub_time = match.group()

        return {
            'url': article_url,
            'title': title,
            'publish_time': pub_time,
            'content': text,
            'comments': []   # 暂留位置，后续填充评论
        }
    except Exception as e:
        print(f"提取新闻正文失败 {article_url}: {e}")
        return None


# ================== 评论提取 ==================
def fetch_comments(article_id: str, page: int = 1, page_size: int = 20) -> List[Dict]:
    url = f"https://example.com/api/comment/list?articleId={article_id}&page={page}&size={page_size}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
        comments = []
        # TODO: 根据实际JSON结构解析
        # 假设返回格式为 { "code": 0, "data": { "list": [{"nickname": "xx", "content": "yy", "time": "zz"}] } }
        if data.get('code') == 0 and 'data' in data:
            for item in data['data'].get('list', []):
                comments.append({
                    'author': item.get('nickname', '匿名'),
                    'content': item.get('content', ''),
                    'time': item.get('time', ''),
                })
        return comments
    except Exception as e:
        print(f"获取评论失败 (文章 {article_id}, 页码 {page}): {e}")
        return []


# ================== 主流程 ==================
def main():
    # 1. 目标列表页（可按需修改为其他频道，如 news.cnwest.com/szyw/）
    target_lists = [
        "http://news.cnwest.com/sxxw/",     # 陕西新闻频道
        "http://news.cnwest.com/szyw/",     # 陕西要闻频道
    ]
    all_article_urls = []
    for list_url in target_lists:
        urls = get_article_urls_from_list(list_url, max_articles=30)
        all_article_urls.extend(urls)
        time.sleep(REQUEST_DELAY)

    # 2. 提取新闻正文
    news_data = []
    for idx, url in enumerate(all_article_urls, 1):
        print(f"处理第 {idx}/{len(all_article_urls)} 篇: {url}")
        article = extract_news_text(url)
        if article:
            # 从 URL 中提取文章 ID
            match = re.search(r'/(\d+)\.html$', url)
            article_id = match.group(1) if match else ''
            if article_id:
                # 获取评论
                article['comments'] = fetch_comments(article_id, page=1, page_size=20)
            news_data.append(article)
        time.sleep(REQUEST_DELAY)

    # 3. 保存为 CSV
    with open('western_news_comments.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['新闻标题', '发布时间', '新闻正文', '评论内容', '评论作者', '评论时间'])
        for article in news_data:
            if not article['comments']:
                writer.writerow([article['title'], article['publish_time'], article['content'], '', '', ''])
            else:
                for cmt in article['comments']:
                    writer.writerow([
                        article['title'],
                        article['publish_time'],
                        article['content'],
                        cmt.get('content', ''),
                        cmt.get('author', ''),
                        cmt.get('time', '')
                    ])

    print(f"\n爬取完成！共处理 {len(news_data)} 篇新闻，数据已保存至 western_news_comments.csv")


if __name__ == "__main__":
    main()