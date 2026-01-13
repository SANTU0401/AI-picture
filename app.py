import streamlit as st
import replicate
import time
import requests
import zipfile
import io
from PIL import Image
from replicate.exceptions import ReplicateError

# --- 页面配置 ---
st.set_page_config(page_title="二次元转3D专用工作台", layout="wide")
st.title("🖥️ 二次元转 3D 游戏质感工作台 (ControlNet)")
st.markdown("ℹ️ **核心功能**：专门用于将 Anime/漫画 转换为 3D CGI/虚幻引擎风格，同时**完美保留原图构图和表情**。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("🔑 密钥设置")
    raw_token = st.text_input("Replicate API Token", type="password", help="r8_开头")
    api_token = raw_token.strip() if raw_token else None

    st.divider()
    
    st.header("🎮 3D化参数控制")
    # 针对你想要的效果，我预设了最佳参数
    control_scale = st.slider(
        "线稿锁死程度 (Control Strength)", 
        0.0, 1.5, 0.75, 
        help="推荐 0.75。数值越高，越严格遵守原图线条；数值太低，脸可能会变。"
    )
    
    prompt_strength = st.slider(
        "3D化 程度 (Denoising Strength)", 
        0.1, 1.0, 0.85, 
        help="推荐 0.85。必须够高才能把二次元彻底洗成3D。"
    )
    
    # 增强提示词开关
    use_3d_prompt = st.checkbox("✅ 强制开启 3D 增强咒语", value=True, help="自动加入 Unreal Engine 5, Ray Tracing 等关键词")

# --- 工具函数 ---
def preprocess_image(file_obj):
    """清洗图片格式"""
    try:
        image = Image.open(file_obj)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        return buf
    except Exception as e:
        raise Exception(f"图片处理失败: {e}")

def run_replicate_dynamic(model_name, input_data, token):
    """API 调用函数"""
    client = replicate.Client(api_token=token)
    try:
        model = client.models.get(model_name)
        latest_version = model.latest_version
    except Exception as e:
        raise Exception(f"模型连接失败: {e}")

    for attempt in range(3):
        try:
            prediction = client.predictions.create(version=latest_version, input=input_data)
            prediction.wait()
            if prediction.status == "succeeded": return prediction.output
            elif prediction.status == "failed": 
                if prediction.error and "NSFW" in str(prediction.error): raise Exception("NSFW_ERROR")
                raise Exception(f"生成失败: {prediction.error}")
        except Exception as e:
            if str(e) == "NSFW_ERROR": raise e
            if "429" in str(e):
                st.toast(f"⏳ 限流冷却中... {10 + attempt * 5}s")
                time.sleep(10 + attempt * 5)
                continue
            raise e
    raise Exception("重试超时")

def download_image(url):
    return requests.get(url).content

# --- 主界面 ---
left, right = st.columns([1, 1.5], gap="large")

# 左侧：上传二次元原图
with left:
    st.header("1️⃣ 上传二次元原图")
    # 这里我们不需要“参考风格图”了，因为风格已经硬编码为 3D 真实风
    ref_file = st.file_uploader("上传图片", type=['jpg', 'png'], key="ref")
    if ref_file:
        st.image(ref_file, caption="原图", use_container_width=True)

# 右侧：执行转换
with right:
    st.header("2️⃣ 3D 转换结果")
    
    if ref_file and api_token:
        if st.button("🚀 立即转换为 3D 游戏风格"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.info("正在分析图片内容...")
                clean_img = preprocess_image(ref_file)
                
                # 1. 识别内容 (是个男孩？女孩？)
                content_desc = run_replicate_dynamic(
                    "salesforce/blip", 
                    {"image": clean_img, "task": "image_captioning"}, 
                    api_token
                )
                # 清洗掉 anime 等词，防止 AI 被带偏
                content_clean = content_desc.replace("cartoon", "").replace("anime", "").replace("drawing", "").strip()
                
                # 2. 构建超级 3D 提示词
                # 这是实现你想要效果的关键！
                if use_3d_prompt:
                    final_prompt = (
                        f"hyper-realistic 3d render of {content_clean}, "
                        "unreal engine 5 style, cinematic lighting, ray tracing, "
                        "highly detailed texture, skin pores, 8k resolution, masterpiece, "
                        "CGI, shallow depth of field, photorealistic"
                    )
                else:
                    final_prompt = f"{content_clean}, 3d render, best quality"

                # 强力负面提示词，禁止二次元
                negative_prompt = "anime, cartoon, 2d, sketch, drawing, illustration, painting, flat color, low quality, bad anatomy"

                status_text.info("正在渲染 3D 效果 (ControlNet)...")
                
                # 3. 调用 ControlNet 模型
                output = run_replicate_dynamic(
                    "xiankgx/sdxl-controlnet-canny", 
                    {
                        "image": clean_img,
                        "prompt": final_prompt,
                        "negative_prompt": negative_prompt,
                        "controlnet_conditioning_scale": control_scale, # 锁死线稿
                        "prompt_strength": prompt_strength,             # 风格重绘幅度 (必须高)
                        "num_inference_steps": 40,                      # 步数高一点，质感更好
                        "guidance_scale": 7.5
                    },
                    api_token
                )
                
                img_url = output[0] if isinstance(output, list) else output
                
                # 展示结果
                st.image(img_url, caption="3D 转换结果", use_container_width=True)
                st.markdown(f"**使用的提示词:** `{final_prompt}`")
                st.markdown(f"[下载大图]({img_url})")
                
                status_text.success("✅ 转换完成！")
                progress_bar.progress(1.0)
                
            except Exception as e:
                st.error(f"处理失败: {e}")

    elif not api_token:
        st.warning("👈 请输入 Token")
