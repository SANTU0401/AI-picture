import streamlit as st
import replicate
import os
from PIL import Image
import io
import zipfile

# 页面配置
st.set_page_config(page_title="AI批量风格迁移工具", layout="wide")

st.title("🎨 AI图片风格提取与批量生成工具")
st.markdown("上传参考图提取风格 -> 上传批量图片 -> AI自动应用风格")

# --- 侧边栏：设置 ---
with st.sidebar:
    st.header("🔑 设置")
    api_token = st.text_input("输入 Replicate API Token", type="password", help="请从 replicate.com 获取")
    
    st.header("⚙️ 生成参数")
    strength = st.slider("风格重绘幅度 (Strength)", 0.1, 1.0, 0.75, help="值越大，越像参考风格；值越小，越像原图")
    num_steps = st.slider("生成步数", 20, 50, 30)

# 初始化 Replicate 客户端
if api_token:
    os.environ["REPLICATE_API_TOKEN"] = api_token

# --- 步骤 1: 上传与分析 ---
st.subheader("1. 上传参考风格图")
ref_file = st.file_uploader("上传一张包含你想要风格的图片", type=['png', 'jpg', 'jpeg'])

prompt_text = ""
if ref_file and api_token:
    st.image(ref_file, caption="参考风格图", width=300)
    
    if st.button("🔍 分析风格提示词 (提取Prompt)"):
        with st.spinner("AI正在观察图片并提取风格关键词..."):
            try:
                # 使用 CLIP Interrogator 模型反推提示词
                output = replicate.run(
                    "pharmapsychotic/clip-interrogator:a24998d0ddb2eabd20197e9e38ef2049d59e99dd94ca9e87900408cb837130b0",
                    input={"image": ref_file, "mode": "fast"}
                )
                st.session_state['style_prompt'] = output
                st.success("风格提取成功！")
            except Exception as e:
                st.error(f"发生错误: {str(e)}")

# 显示并允许修改提示词
if 'style_prompt' in st.session_state:
    st.markdown("### 📝 风格提示词 (Style Prompt)")
    style_prompt = st.text_area("AI生成的风格描述 (可手动修改)", st.session_state['style_prompt'], height=100)
else:
    style_prompt = ""

st.markdown("---")

# --- 步骤 2: 批量处理 ---
st.subheader("2. 批量上传内容图并生成")
uploaded_files = st.file_uploader("选择多张需要处理的图片", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files and style_prompt and api_token:
    if st.button(f"🚀 开始批量生成 ({len(uploaded_files)} 张)"):
        
        # 准备结果展示区
        results_container = st.container()
        generated_images = []
        
        progress_bar = st.progress(0)
        
        for idx, img_file in enumerate(uploaded_files):
            with st.spinner(f"正在处理第 {idx+1}/{len(uploaded_files)} 张图片..."):
                try:
                    # 组合提示词：风格 + 基础质量词
                    final_prompt = f"{style_prompt}, high quality, high resolution, 4k"
                    
                    # 调用 SDXL Image-to-Image
                    output = replicate.run(
                        "stability-ai/sdxl:39ed52f2a78e934b3ba6e399ea1a963986eeac40ef080b697b0803a6466b717c",
                        input={
                            "image": img_file,
                            "prompt": final_prompt,
                            "prompt_strength": 1.0 - strength, # Replicate的参数逻辑相反，需要转换
                            "num_inference_steps": num_steps
                        }
                    )
                    
                    # 获取结果URL (Replicate通常返回列表，取第一张)
                    image_url = output[0]
                    generated_images.append((img_file.name, image_url))
                    
                except Exception as e:
                    st.error(f"图片 {img_file.name} 处理失败: {e}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # --- 展示结果 ---
        st.success("✅ 所有图片处理完成！")
        
        # 预览
        cols = st.columns(3)
        for i, (name, url) in enumerate(generated_images):
            with cols[i % 3]:
                st.image(url, caption=f"Result: {name}")
                st.markdown(f"[下载图片]({url})")

else:
    if not api_token:
        st.info("👈 请先在左侧输入 Replicate API Token")
    elif not ref_file:
        st.info("👆 请先上传参考图")
    elif not 'style_prompt' in st.session_state:
        st.info("👆 请点击“分析风格提示词”")
