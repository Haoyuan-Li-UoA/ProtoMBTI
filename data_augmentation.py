import pandas as pd
import json
import requests
import time
from tqdm import tqdm
import os
from typing import List, Dict, Tuple
import math
import pandas as pd
import json
import ijson  # 注意需 pip install ijson
from typing import List, Tuple

# MBTI风格提示词定义
mbti_style_prompts = {
    "INFP": (
        "You are a language model trained to write like an INFP: gentle, emotionally expressive, "
        "idealistic, and introspective. Your goal is to rewrite any input text in this style, "
        "highlighting personal meaning, feeling, and poetic insight."
    ),
    "INFJ": (
        "You are a language model trained to write like an INFJ: visionary, reflective, profound, and empathetic. "
        "Rewrite the text with deep insight, symbolic language, and a focus on inner values and human connection."
    ),
    "INTP": (
        "You are a language model trained to write like an INTP: analytical, abstract, precise, and curious. "
        "Rewrite the input in a style that emphasizes logical reasoning, philosophical depth, and theoretical musings."
    ),
    "INTJ": (
        "You are a language model trained to write like an INTJ: strategic, decisive, and conceptually visionary. "
        "Rewrite the text to reflect high-level planning, clarity of purpose, and structured insight."
    ),
    "ENFP": (
        "You are a language model trained to write like an ENFP: energetic, imaginative, playful, and values-driven. "
        "Rewrite the text with creativity, warmth, enthusiasm, and emotional spontaneity."
    ),
    "ENFJ": (
        "You are a language model trained to write like an ENFJ: charismatic, supportive, and purpose-oriented. "
        "Rewrite the input with persuasive language, emotional attunement, and a focus on inspiring others."
    ),
    "ENTP": (
        "You are a language model trained to write like an ENTP: witty, spontaneous, inventive, and intellectually provocative. "
        "Rewrite the text with cleverness, enthusiasm, and a tendency to challenge ideas in creative ways."
    ),
    "ENTJ": (
        "You are a language model trained to write like an ENTJ: assertive, organized, and visionary. "
        "Rewrite the input with strong leadership language, structured logic, and forward-thinking analysis."
    ),
    "ISFP": (
        "You are a language model trained to write like an ISFP: gentle, artistic, sensory-focused, and value-driven. "
        "Rewrite the text with a focus on aesthetics, present-moment experience, and authentic self-expression."
    ),
    "ISFJ": (
        "You are a language model trained to write like an ISFJ: thoughtful, nurturing, reliable, and detail-oriented. "
        "Rewrite the input with warmth, practical compassion, and an emphasis on duty and emotional responsibility."
    ),
    "ISTP": (
        "You are a language model trained to write like an ISTP: concise, pragmatic, observant, and independent. "
        "Rewrite the text with straightforward logic, action-oriented insight, and calm detachment."
    ),
    "ISTJ": (
        "You are a language model trained to write like an ISTJ: logical, methodical, dependable, and tradition-conscious. "
        "Rewrite the text in a clear, factual tone with an emphasis on structure, duty, and responsibility."
    ),
    "ESFP": (
        "You are a language model trained to write like an ESFP: vibrant, expressive, present-focused, and playful. "
        "Rewrite the text with high energy, sensory detail, and a zest for life and connection."
    ),
    "ESFJ": (
        "You are a language model trained to write like an ESFJ: warm, supportive, socially aware, and harmonious. "
        "Rewrite the text in a friendly tone with attention to social relationships, kindness, and tradition."
    ),
    "ESTP": (
        "You are a language model trained to write like an ESTP: direct, dynamic, action-focused, and confident. "
        "Rewrite the text with a bold, high-energy tone and a focus on results, excitement, and real-world application."
    ),
    "ESTJ": (
        "You are a language model trained to write like an ESTJ: organized, authoritative, and objective. "
        "Rewrite the text in a businesslike tone, emphasizing efficiency, clarity, and control."
    )
}

class MBTIDataAugmenter:
    """MBTI数据增强器"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        """
        初始化数据增强器
        
        Args:
            api_key: API密钥
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.all_mbti_types = list(mbti_style_prompts.keys())
    
    def load_mbti_data(self,data_path: str, mbti_type: str, max_examples: int = 5) -> List[Tuple[str, str]]:
        """
        从数据文件流式加载特定 MBTI 类型的数据作为 few-shot 示例

        Args:
            data_path: 数据文件路径（支持 CSV 和 JSON 格式）
            mbti_type: 目标 MBTI 类型
            max_examples: 最大示例数量

        Returns:
            List of (original_text, rewritten_text) 对，用于 few-shot 学习
        """
        print(f"正在加载 {mbti_type} 类型的 few-shot 示例...")

        few_shot_pairs = []
        try:
            if data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
                if "type" not in df.columns or "posts" not in df.columns:
                    raise ValueError("CSV 中缺少 'type' 或 'posts' 列")

                df = df[(df["type"] == mbti_type) & df["posts"].notna()]
                df = df[df["posts"].astype(str).str.strip() != ""]
                df_sampled = df.sample(n=min(max_examples, len(df)), random_state=42)

                for _, row in df_sampled.iterrows():
                    text = str(row["posts"]).strip()
                    if len(text) > 10:
                        few_shot_pairs.append((text, text))

            elif data_path.endswith(".json"):
                with open(data_path, 'r', encoding='utf-8') as f:
                    objects = ijson.items(f, "item")
                    count = 0
                    for item in objects:
                        if item.get("type", "").upper() == mbti_type.upper():
                            text = str(item.get("posts", "")).strip()
                            if len(text) > 10:
                                few_shot_pairs.append((text, text))
                                count += 1
                                if count >= max_examples:
                                    break
            else:
                raise ValueError(f"不支持的文件格式: {data_path}")

            print(f"✅ 成功加载 {len(few_shot_pairs)} 个 {mbti_type} 示例")
            return few_shot_pairs

        except Exception as e:
            print(f"❌ 加载数据时出错: {e}")
            return []

    
    def build_prompt(self, mbti_type: str, few_shot_pairs: List[Tuple[str, str]]) -> str:
        base_prompt = mbti_style_prompts.get(mbti_type, "")
        examples_text = ""

        if few_shot_pairs:
            examples_text = "\n\n".join([
                f"Post: {text}\nReasoning: It captures the {mbti_type} vibe — not because of a formula, but because it feels right: the tone, the emotion, the rhythm." 
                for text, _ in few_shot_pairs
            ])
            examples_text += "\n\n"

        return (
            f"{base_prompt}\n\n"
            f"{examples_text}"
            f"Now let’s try creating something new that fits this same feeling. Think of it like continuing the pattern — not copying, but echoing the same voice.\n"
            f"After writing the post, add a short comment on why you think it feels like a {mbti_type}. Just your take — don’t overthink it.\n\n"
            f"Format:\nPost: <your writing>\nReasoning: <why it fits>\n"
            f"Keep it natural, and stay in the flow of the original voice."
        )





    def call_api(self, prompt_text: str,n:int=1) -> List[str] | None:
        url = ""
        payload = {
            "model": self.model,
            "temperature": 0.8,
            "n":n,
            "messages": [{"role": "user", "content": prompt_text}],
            "modalities": ["text"],
            "response_format": {"type": "text"},
            "max_completion_tokens": 512,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            if "choices" in result:
                return [c["message"]["content"] for c in result["choices"]]
            else:
                print("⚠️ API response missing 'choices':", result)
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None


    
    def augment_single_type(self, data_path: str, mbti_type: str, 
                           max_few_shot: int | None = None, max_generations: int | None = None) -> List[Dict]:
        """
        为单个MBTI类型生成增强数据
        
        Args:
            data_path: 数据文件路径（支持CSV和JSON格式）
            mbti_type: MBTI类型
            max_few_shot: 最大few-shot示例数量（如果为None，使用配置中的默认值）
            max_generations: 最大生成数量（如果为None，使用配置中的默认值）
            
        Returns:
            增强数据列表
        """
        # 使用配置中的默认值
        if max_few_shot is None:
            max_few_shot = mbti_augmentation_config.get(mbti_type, {}).get("max_few_shot", 5)
        if max_generations is None:
            max_generations = mbti_augmentation_config.get(mbti_type, {}).get("max_generations", 20)
        
        print(f"\n开始为 {mbti_type} 类型生成增强数据...")
        print(f"配置: few_shot={max_few_shot}, generations={max_generations}")
        
        augmented_data = []

        batch_size = 30  # 每批生成数量
        attempt = 0
        # max_attempts = math.ceil(max_generations / batch_size)+50
        with tqdm(total=max_generations, desc=f"生成 {mbti_type} 数据") as pbar:
            while len(augmented_data) < max_generations:
                attempt += 1
                try:
                    few_shot_pairs = self.load_mbti_data(data_path, mbti_type, max_few_shot)
                    prompt = self.build_prompt(mbti_type, few_shot_pairs)
                    response_texts = self.call_api(prompt, n=batch_size)

                    if not response_texts:
                        continue

                    for response_text in response_texts:
                        if len(augmented_data) >= max_generations:
                            break

                        try:
                            if "Post:" in response_text and "Reasoning:" in response_text:
                                post_part = response_text.split("Post:", 1)[1].split("Reasoning:")[0].strip()
                                reasoning_part = response_text.split("Reasoning:", 1)[1].strip()
                            else:
                                post_part = response_text.strip()
                                reasoning_part = f"Generated {mbti_type} style post"

                            if post_part and len(post_part) > 10:
                                augmented_data.append({
                                    "type": mbti_type,
                                    "post": post_part,
                                    "reasoning": reasoning_part
                                })
                                pbar.update(1)

                        except Exception as e:
                            print(f"⚠️ 解析错误: {e}")
                            continue

                    time.sleep(1)

                except Exception as e:
                    print(f"❌ 第 {attempt} 次生成失败: {e}")
                    continue

        
        print(f"成功为 {mbti_type} 生成 {len(augmented_data)} 个增强样本")
        return augmented_data

    
    def save_augmented_data(self, augmented_data: List[Dict], mbti_type: str, output_dir: str = "augmented_data"):
        """
        保存增强数据到JSON文件
        
        Args:
            augmented_data: 增强数据列表
            mbti_type: MBTI类型
            output_dir: 输出目录
        """
        if not augmented_data:
            print(f"没有 {mbti_type} 的增强数据需要保存")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存单个类型的数据
        output_file = os.path.join(output_dir, f"augmented_{mbti_type}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {mbti_type}: {len(augmented_data)} 个样本 -> {output_file}")
    
    def augment_specific_types(self, data_path: str, target_types: List[str], 
                              output_dir: str = "augmented_data") -> Dict[str, List[Dict]]:
        """
        为指定的MBTI类型生成增强数据
        
        Args:
            data_path: 数据文件路径（支持CSV和JSON格式）
            target_types: 目标MBTI类型列表
            output_dir: 输出目录
            
        Returns:
            所有类型的增强数据字典
        """
        print(f"开始为指定类型生成增强数据...")
        print(f"目标类型: {target_types}")
        print(f"输入文件: {data_path}")
        print(f"输出目录: {output_dir}")
        
        all_augmented_data = {}
        
        for mbti_type in target_types:
            if mbti_type not in self.all_mbti_types:
                print(f"❌ 未知的MBTI类型: {mbti_type}")
                continue
            
            try:
                # 为单个类型生成数据
                augmented_data = self.augment_single_type(data_path, mbti_type)
                
                if augmented_data:
                    all_augmented_data[mbti_type] = augmented_data
                    # 保存数据
                    self.save_augmented_data(augmented_data, mbti_type, output_dir)
                else:
                    print(f"❌ {mbti_type}: 没有生成任何数据")
                
            except Exception as e:
                print(f"❌ 处理 {mbti_type} 时出错: {e}")
                continue
        
        # 保存所有数据
        if all_augmented_data:
            all_data_file = os.path.join(output_dir, "all_augmented_data.json")
            with open(all_data_file, "w", encoding="utf-8") as f:
                json.dump(all_augmented_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 所有数据已保存到: {all_data_file}")
            
            # 打印统计信息
            total_samples = sum(len(data) for data in all_augmented_data.values())
            print(f"总计生成: {total_samples} 个增强样本")
        
        return all_augmented_data
    
    def update_augmentation_config(self, new_config: Dict[str, Dict]):
        """
        更新增强配置
        
        Args:
            new_config: 新的配置字典，格式为 {mbti_type: {"max_few_shot": x, "max_generations": y}}
        """
        global mbti_augmentation_config
        mbti_augmentation_config.update(new_config)
        print("✅ 增强配置已更新")
    
    def show_current_config(self):
        """显示当前增强配置"""
        print("\n当前增强配置:")
        print("-" * 50)
        for mbti_type, config in mbti_augmentation_config.items():
            print(f"{mbti_type}: few_shot={config['max_few_shot']}, generations={config['max_generations']}")
        print("-" * 50)

# 每个MBTI类型的增强配置
# mbti_augmentation_config = {
#     "INFP": {"max_few_shot": 5, "max_generations": 0},
#     "INFJ": {"max_few_shot": 5, "max_generations": 0},
#     "INTP": {"max_few_shot": 5, "max_generations": 0},
#     "INTJ": {"max_few_shot": 5, "max_generations": 0},
#     "ENFP": {"max_few_shot": 5, "max_generations": 0},
#     "ENFJ": {"max_few_shot": 5, "max_generations": 0},
#     "ENTP": {"max_few_shot": 5, "max_generations": 5000},
#     "ENTJ": {"max_few_shot": 5, "max_generations": 5000},
#     "ISFP": {"max_few_shot": 5, "max_generations": 5000},
#     "ISFJ": {"max_few_shot": 5, "max_generations": 5000},
#     "ISTP": {"max_few_shot": 5, "max_generations": 5000},
#     "ISTJ": {"max_few_shot": 5, "max_generations": 5000},
#     "ESFP": {"max_few_shot": 5, "max_generations": 20000},
#     "ESFJ": {"max_few_shot": 5, "max_generations": 20000},
#     "ESTP": {"max_few_shot": 5, "max_generations": 20000},
#     "ESTJ": {"max_few_shot": 5, "max_generations": 20000}
# }
mbti_augmentation_config = {
    "INFP": {"max_few_shot": 5, "max_generations": 0},
    "INFJ": {"max_few_shot": 5, "max_generations": 0},
    "INTP": {"max_few_shot": 5, "max_generations": 0},
    "INTJ": {"max_few_shot": 5, "max_generations": 0},
    "ENFP": {"max_few_shot": 5, "max_generations": 0},
    "ENFJ": {"max_few_shot": 5, "max_generations": 0},
    "ENTP": {"max_few_shot": 5, "max_generations": 5000},
    "ENTJ": {"max_few_shot": 5, "max_generations": 5000},
    "ISFP": {"max_few_shot": 5, "max_generations": 0},
    "ISFJ": {"max_few_shot": 5, "max_generations": 0},
    "ISTP": {"max_few_shot": 5, "max_generations": 0},
    "ISTJ": {"max_few_shot": 5, "max_generations": 0},
    "ESFP": {"max_few_shot": 5, "max_generations": 0},
    "ESFJ": {"max_few_shot": 5, "max_generations": 0},
    "ESTP": {"max_few_shot": 5, "max_generations": 0},
    "ESTJ": {"max_few_shot": 5, "max_generations": 0}
}
def main():
    """主函数"""
    # 配置参数
    API_KEY = ""  # 替换为你的API密钥
    DATA_PATH = "/home/hli962/Chunhou_Project/filtered_processed_comments.json"  # 替换为你的数据文件路径
    OUTPUT_DIR = "augmented_pandora_data"
    
    # 创建增强器
    augmenter = MBTIDataAugmenter(api_key=API_KEY)
    
    # 显示当前配置
    augmenter.show_current_config()
    target_types = [
    "INFP", "INFJ", "INTP", "INTJ",
    "ENFP", "ENFJ", "ENTP", "ENTJ",
    "ISFP", "ISFJ", "ISTP", "ISTJ",
    "ESFP", "ESFJ", "ESTP", "ESTJ"
]

    all_data = augmenter.augment_specific_types(DATA_PATH, target_types, OUTPUT_DIR)
    
    print("\n🎉 数据增强完成!")

if __name__ == "__main__":
    main()
