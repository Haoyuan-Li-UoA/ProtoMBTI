import json
import time
import requests
import re
import random
import os
from tqdm import tqdm
import concurrent.futures
import threading
from typing import List, Dict, Optional, Tuple

# ===== API Wrapper =====
class LLM_API_Wrapper:
    def __init__(self, model, api_key, base_url=""):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()  # 复用连接
        
    def call_api_batch(self, prompts: List[str], timeout_s: int = 90) -> List[Optional[str]]:
        """批量调用API"""
        results = [None] * len(prompts)
        
        def process_single(idx_prompt):
            idx, prompt = idx_prompt
            try:
                payload = {
                    "model": self.model,
                    "temperature": 0.6,
                    "messages": [{"role": "user", "content": prompt}],
                    "modalities": ["text"],
                    "response_format": {"type": "text"},
                    "max_completion_tokens": 512,
                    "stream": False
                }
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                resp = self.session.post(self.base_url, headers=headers, json=payload, timeout=timeout_s)
                resp.raise_for_status()
                data = resp.json()
                
                if "choices" in data and data["choices"]:
                    results[idx] = data["choices"][0]["message"]["content"]
                    return True
                else:
                    print(f"⚠️ Missing 'choices' in response for item {idx}")
                    return False
                    
            except Exception as e:
                print(f"❌ Request failed for item {idx}: {e}")
                return False
        
        # 使用线程池并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_single, (i, prompt)) for i, prompt in enumerate(prompts)]
            concurrent.futures.wait(futures)
        
        return results

# ===== Prompt Builders =====
def build_full_analysis_prompt(text: str) -> str:
    return f"""
You are a psycholinguistics expert. Analyze the following social media post from three perspectives:

1) Semantic Summary: main idea or intention.
2) Sentiment Analysis: emotions/attitudes.
3) Linguistic Style: writing style (e.g., emotional, rational, informal, formal, vague).

Return ONLY valid JSON with the exact keys below and no extra text:

{{
  "semantic_view": "...",
  "sentiment_view": "...",
  "linguistic_view": "..."
}}

Post:
\"\"\"{(text or '').strip()[:1024]}\"\"\"""".strip()

def build_fallback_prompt(text: str) -> str:
    return f"""
Provide a STRICT JSON object with three short fields summarizing the post:

{{
  "semantic_view": "<1-2 sentences>",
  "sentiment_view": "<one or two emotions>",
  "linguistic_view": "<style words>"
}}

Post: {(text or '').strip()[:1024]}""".strip()

# ===== JSON Parser =====
def safe_extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    txt = raw.strip()

    # 去掉代码块标记
    if txt.startswith("```"):
        txt = re.sub(r"^```(json)?", "", txt, flags=re.IGNORECASE).strip()
        if txt.endswith("```"):
            txt = txt[:-3].strip()

    # 抽取第一个 {...}
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        txt = m.group(0)

    # 若全是单引号，尝试替换为双引号
    if txt.count('"') == 0 and txt.count("'") > 0:
        txt = txt.replace("'", "'").replace("'", "'")
        txt = re.sub(r"(?<!\\)'", '"', txt)

    # 去掉尾逗号
    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)

    try:
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            return None
        for k in ("semantic_view", "sentiment_view", "linguistic_view"):
            if k not in obj or not isinstance(obj[k], str):
                return None
        return obj
    except Exception:
        return None

# ===== 兜底生成 =====
def fallback_views_from_text(text: str) -> dict:
    text = (text or "").strip()
    # 语义：一句话或前180字符
    semantic = ""
    if text:
        sentence = re.split(r"(?<=[.!?。！？])\s+", text)[0]
        semantic = sentence[:180]
    if not semantic:
        semantic = "The post contains limited context; the main idea is unclear."

    lowered = text.lower()
    if any(w in lowered for w in ["love", "great", "happy", "excited", "enjoy", "喜欢", "高兴", "开心"]):
        senti = "Positive, optimistic."
    elif any(w in lowered for w in ["hate", "angry", "sad", "tired", "annoyed", "讨厌", "生气", "难过", "疲惫"]):
        senti = "Negative, possibly frustrated or tired."
    else:
        senti = "Neutral or mixed."

    style_tokens = []
    if re.search(r"[A-Z]{3,}", text): style_tokens.append("emphatic")
    if re.search(r"[!！]{1,}", text): style_tokens.append("expressive")
    if re.search(r":[)D]|😂|🤣|😅|🙂|😉", text): style_tokens.append("informal")
    if not style_tokens: style_tokens = ["conversational"]
    ling = ", ".join(style_tokens)

    return {
        "semantic_view": semantic,
        "sentiment_view": senti,
        "linguistic_view": ling
    }

# ===== 批量处理函数 =====
def process_batch(llm: LLM_API_Wrapper, batch_data: List[Tuple[int, dict]], 
                 max_retry: int = 3) -> List[Tuple[int, dict]]:
    """处理一批数据"""
    
    # 准备批量prompts
    prompts = []
    indices = []
    
    for idx, item in batch_data:
        post = item.get("posts") or item.get("posts_cleaned") or ""
        prompt = build_full_analysis_prompt(post)
        prompts.append(prompt)
        indices.append(idx)
    
    results = []
    failed_items = []
    
    for attempt in range(max_retry):
        if not prompts:  # 所有都成功了
            break
            
        print(f"🔄 Batch attempt {attempt + 1}/{max_retry}, processing {len(prompts)} items...")
        
        # 批量调用API
        api_results = llm.call_api_batch(prompts)
        
        new_prompts = []
        new_indices = []
        
        for i, (result, idx) in enumerate(zip(api_results, indices)):
            # 找到对应的原始item
            original_item = None
            for orig_idx, orig_item in batch_data:
                if orig_idx == idx:
                    original_item = orig_item
                    break
            
            if original_item is None:
                continue
            
            if result:
                parsed = safe_extract_json(result)
                if parsed and all(isinstance(parsed[k], str) and parsed[k].strip() for k in parsed):
                    # 成功解析
                    original_item["semantic_view"] = parsed["semantic_view"]
                    original_item["sentiment_view"] = parsed["sentiment_view"]
                    original_item["linguistic_view"] = parsed["linguistic_view"]
                    results.append((idx, original_item))
                    continue
            
            # 失败的情况
            if attempt == 0:
                # 第一次失败，尝试简化prompt
                post = original_item.get("posts") or original_item.get("posts_cleaned") or ""
                new_prompts.append(build_fallback_prompt(post))
                new_indices.append(idx)
            else:
                # 多次失败，加入待兜底列表
                failed_items.append((idx, original_item))
        
        prompts = new_prompts
        indices = new_indices
        
        if prompts and attempt < max_retry - 1:
            # 重试前等待
            sleep_time = 2 * (attempt + 1) + random.uniform(0, 1)
            time.sleep(sleep_time)
    
    # 处理最终失败的items
    for idx, item in failed_items:
        print(f"❗ Using local fallback for item {idx}")
        post = item.get("posts") or item.get("posts_cleaned") or ""
        views = fallback_views_from_text(post)
        item["semantic_view"] = views["semantic_view"]
        item["sentiment_view"] = views["sentiment_view"]
        item["linguistic_view"] = views["linguistic_view"]
        results.append((idx, item))
    
    return results

# ===== 主流程配置 =====
MODEL_NAME = "gpt-4.1-mini"
API_KEY = ""
INPUT_FILE = "sampled_3000_per_type.json"
OUTPUT_FILE = "mbti_sample_with_all_views_pandora.json"
BATCH_SIZE = 20  # 批量大小
MAX_RETRY = 3
SAVE_INTERVAL = 5  # 每处理几个批次保存一次

# ===== Init =====
llm = LLM_API_Wrapper(model=MODEL_NAME, api_key=API_KEY)

# ===== Load data =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    full_data = json.load(f)

print(f"📦 Loaded {len(full_data)} samples. Running optimized batch processing...")

# ===== 处理现有输出 =====
existing = []
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"🔁 Found existing output with {len(existing)} items.")
    except Exception:
        existing = []

# 确保目标列表和原始一样长
if not existing or len(existing) != len(full_data):
    selected_samples = [item.copy() for item in full_data]
else:
    selected_samples = existing

# ===== 找到开始位置 =====
start_idx = 0
for i in range(len(selected_samples)):
    item = selected_samples[i]
    if all(isinstance(item.get(k, ""), str) and item.get(k, "").strip() for k in
           ("semantic_view", "sentiment_view", "linguistic_view")):
        start_idx = i + 1
    else:
        break

print(f"🚀 Starting from index {start_idx}")

# ===== 批量处理 =====
total_items = len(selected_samples) - start_idx
processed_count = 0
failed_total = 0

for batch_start in tqdm(range(start_idx, len(selected_samples), BATCH_SIZE), 
                       desc="Processing batches"):
    batch_end = min(batch_start + BATCH_SIZE, len(selected_samples))
    
    # 准备当前批次数据
    batch_data = [(i, selected_samples[i]) for i in range(batch_start, batch_end)]
    
    # 处理批次
    batch_results = process_batch(llm, batch_data, MAX_RETRY)
    batch_num = (batch_start - start_idx) // BATCH_SIZE + 1
    # 打印批次输出示例
    print(f"\n📋 批次 {batch_num} 输出示例:")
    for i, (idx, updated_item) in enumerate(batch_results[:2]):  # 只显示前2个
        post_preview = (updated_item.get("posts") or updated_item.get("posts_cleaned") or "")[:100]
        print(f"  样本 {idx}:")
        print(f"    原文: {post_preview}...")
        print(f"    语义: {updated_item.get('semantic_view', '')[:80]}...")
        print(f"    情感: {updated_item.get('sentiment_view', '')}")
        print(f"    风格: {updated_item.get('linguistic_view', '')}")
        if i < len(batch_results) - 1:
            print()
    
    if len(batch_results) > 2:
        print(f"    ... (还有 {len(batch_results) - 2} 个样本)")
    
    # 更新结果
    for idx, updated_item in batch_results:
        selected_samples[idx] = updated_item
    
    processed_count += len(batch_results)
    
    # 打印当前批次处理进度
    print(f"✅ 批次 {batch_num} 完成: 处理了 {len(batch_results)} 个样本 (总进度: {processed_count}/{total_items})\n")
    
    # 定期保存
    if batch_num % SAVE_INTERVAL == 0:
        print(f"💾 Saving progress... ({processed_count}/{total_items} processed)")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(selected_samples, f, ensure_ascii=False, indent=2)

# ===== 最终保存 =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(selected_samples, f, ensure_ascii=False, indent=2)

print(f"\n✅ 完成！已处理 {processed_count} 个样本")
print(f"📄 结果已保存至 {OUTPUT_FILE}")
if failed_total > 0:
    print(f"⚠️  {failed_total} 个样本使用了本地兜底生成")