import requests
import csv
import time
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'http://news.cnwest.com/',
}
REQUEST_DELAY = 1

def get_article_urls_from_list(list_url: str, max_articles: int = 30) -> List[str]:
    """从列表页提取文章URL，兼容多种链接格式"""
    urls = set()
    try:
        resp = requests.get(list_url, headers=HEADERS, timeout=15)
        print(f"请求列表页 {list_url} 状态码: {resp.status_code}")
        if resp.status_code != 200:
            return []
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 查找所有包含 .html 的链接，且路径包含 /a/年/月/日/数字.html
        for a in soup.find_all('a', href=True):
            href = a['href']
            # 匹配形如 /sxxw/a/2026/04/17/123456.html 或 /a/2026/04/17/123456.html
            if re.search(r'/a/\d{4}/\d{2}/\d{2}/\d+\.html', href):
                full_url = href if href.startswith('http') else f"http://news.cnwest.com{href}"
                urls.add(full_url)
                if len(urls) >= max_articles:
                    break
        print(f"从 {list_url} 提取到 {len(urls)} 个文章链接")
        if len(urls) == 0:
            # 调试：打印前5个a标签的href
            sample_hrefs = [a['href'] for a in soup.find_all('a', href=True)[:5]]
            print(f"样本链接: {sample_hrefs}")
    except Exception as e:
        print(f"列表页解析失败 {list_url}: {e}")
    return list(urls)

def extract_news_text(article_url: str) -> Optional[Dict]:
    """提取单篇新闻的标题、正文、发布时间"""
    try:
        resp = requests.get(article_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"文章页请求失败 {article_url} 状态码: {resp.status_code}")
            return None
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 标题
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ''
        # 去除西部网标题后缀
        title = re.sub(r'\s*-\s*西部网.*$', '', title)
        
        # 正文容器：按常见可能性依次查找
        content_div = None
        for selector in ['div.article-content', 'div.content', 'div.article-text', 
                         'div.main-content', 'article', 'div.article-con']:
            content_div = soup.select_one(selector)
            if content_div:
                break
        
        if content_div:
            # 提取文本，但保留段落结构（用换行分隔）
            paragraphs = content_div.find_all('p')
            if paragraphs:
                text = '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            else:
                text = content_div.get_text(strip=True)
        else:
            # 降级：提取所有p标签
            paragraphs = soup.find_all('p')
            text = '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        
        # 清洗
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 50:
            print(f"正文过短（{len(text)}字符），可能提取失败: {article_url}")
            return None
        
        # 发布时间
        pub_time = ''
        # 尝试meta标签
        time_meta = soup.find('meta', {'name': 'publishdate'}) or soup.find('meta', {'property': 'article:published_time'})
        if time_meta and time_meta.get('content'):
            pub_time = time_meta['content']
        else:
            # 在页面文本中查找日期时间
            date_pattern = r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}'
            match = re.search(date_pattern, soup.get_text())
            if match:
                pub_time = match.group()
        
        return {
            'url': article_url,
            'title': title,
            'publish_time': pub_time,
            'content': text
        }
    except Exception as e:
        print(f"提取正文失败 {article_url}: {e}")
        return None

def main():
    # 可更换为其他频道，如 'http://news.cnwest.com/szyw/'
    target_lists = [
        "http://news.cnwest.com/sxxw/",   # 陕西新闻
        "http://news.cnwest.com/szyw/",   # 陕西要闻
    ]
    
    all_article_urls = []
    for list_url in target_lists:
        urls = get_article_urls_from_list(list_url, max_articles=30)
        all_article_urls.extend(urls)
        time.sleep(REQUEST_DELAY)
    
    print(f"总共获取到 {len(all_article_urls)} 篇文章链接")
    if not all_article_urls:
        print("未获取到任何文章链接，请检查列表页结构或正则表达式。")
        return
    
    news_data = []
    for idx, url in enumerate(all_article_urls, 1):
        print(f"处理第 {idx}/{len(all_article_urls)} 篇: {url}")
        article = extract_news_text(url)
        if article:
            news_data.append(article)
        time.sleep(REQUEST_DELAY)
    
    # 保存为CSV（每条新闻一行，不含评论）
    with open('western_news.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['新闻标题', '发布时间', '新闻正文', '原文链接'])
        for art in news_data:
            writer.writerow([art['title'], art['publish_time'], art['content'], art['url']])
    
    print(f"\n爬取完成！共处理 {len(news_data)} 篇新闻，数据保存至 western_news.csv")

if __name__ == "__main__":
    main()
