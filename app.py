import streamlit as st
import replicate
import time
import requests
import zipfile
import io

# --- 页面基础设置 ---
st.set_page_config(page_title="AI全能风格迁移工作台", layout="wide")
st.title("🎨 AI全能风格迁移工作台 (分析+替换+生成+打包)")

# --- 侧边栏：配置区 ---
with st.sidebar:
    st.header("🔑 核心设置")
    raw_token = st.text_input("Replicate API Token", type="password", help="r8_开头")
    api_token = raw_token.strip() if raw_token else None
    
    if api_token and not api_token.startswith("r8_"):
        st.error("⚠️ Token 格式错误")

    st.header("⚙️ 生成控制")
    strength = st.slider("风格重塑程度", 0.1, 1.0, 0.75, help="1.0为完全重绘，0.5保留更多原图结构")
    st.info("💡 逻辑说明：\n1. AI提取参考图的【风格】\n2. AI提取批量图的【内容】\n3. 两者融合生成新图")

# --- 核心工具函数 ---
def run_replicate(model, input_data, token):
    client = replicate.Client(api_token=token)
    return client.run(model, input=input_data)

# 用于下载生成的图片以便打包
def download_image(url):
    response = requests.get(url)
    return response.content

# --- 第一部分：参考图风格分析 ---
st.header("1️⃣ 参考风格提取 (Style Extraction)")
col1, col2 = st.columns([1, 2])

with col1:
    ref_file = st.file_uploader("上传一张风格参考图", type=['png', 'jpg', 'jpeg'], key="ref")

if ref_file:
    with col1:
        st.image(ref_file, caption="参考图", use_container_width=True)

    with col2:
        if api_token:
            if st.button("🔍 分析风格提示词"):
                with st.spinner("正在使用 CLIP 模型分析画面风格..."):
                    try:
                        # 使用 CLIP Interrogator 分析风格
                        output = run_replicate(
                            "pharmapsychotic/clip-interrogator:8151e1c9f47e696fa316146a2e35812ccf79cfc9eba05b11c7f450155102af70",
                            {"image": ref_file, "mode": "fast"},
                            api_token
                        )
                        st.session_state['style_tags'] = output
                        st.success("风格提取完成！")
                    except Exception as e:
                        st.error(f"分析失败: {e}")
            
            # 允许用户编辑风格词
            style_prompt = st.text_area(
                "风格提示词 (Style Prompts)", 
                value=st.session_state.get('style_tags', ""),
                height=150,
                placeholder="此处将显示AI分析出的风格关键词，例如: oil painting, cyberpunk, lighting..."
            )
else:
    style_prompt = ""

st.markdown("---")

# --- 第二部分：批量内容分析与生成 ---
st.header("2️⃣ 批量融合与生成 (Batch Processing)")
batch_files = st.file_uploader("上传需要处理的批量图片", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="batch")

# 只有当所有条件具备时才显示开始按钮
if batch_files and style_prompt and api_token:
    
    start_btn = st.button(f"🚀 开始全流程处理 ({len(batch_files)} 张图片)")
    
    if start_btn:
        # 初始化存储，用于打包下载
        zip_buffer = io.BytesIO()
        generated_files_count = 0
        
        progress_bar = st.progress(0)
        status_area = st.empty()
        results_container = st.container()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            
            for idx, img_file in enumerate(batch_files):
                current_step_str = f"[{idx+1}/{len(batch_files)}] {img_file.name}"
                
                # --- 阶段 A: 分析当前图片的内容 ---
                status_area.info(f"👁️ 正在识别内容: {current_step_str} ...")
                content_desc = ""
                try:
                    # 使用 BLIP 模型快速识别图片内容 (例如: "a cat sitting on a table")
                    content_output = run_replicate(
                        "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
                        {"image": img_file, "task": "image_captioning"},
                        api_token
                    )
                    # 清理输出，加上 caption: 前缀
                    content_desc = content_output.strip()
                except Exception as e:
                    st.warning(f"内容识别失败，将仅使用风格词。错误: {e}")
                    content_desc = "original image content"

                # --- 阶段 B: 提示词融合 ---
                # 逻辑：风格词 + 内容描述
                final_combined_prompt = f"{style_prompt}, {content_desc}, high quality, 8k"
                
                # --- 阶段 C: 生成图片 ---
                status_area.info(f"🎨 正在绘图: {current_step_str} (内容: {content_desc}) ...")
                
                try:
                    # 使用 SDXL-Lightning 快速生成
                    output_urls = run_replicate(
                        "bytedance/sdxl-lightning-4step:727e49a643e999d602a896c774a0158e63aa74b62784b8d42055368a28ecbd9f",
                        {
                            "image": img_file,
                            "prompt": final_combined_prompt,
                            "prompt_strength": 1.0 - strength, 
                            "num_inference_steps": 4,
                            "guidance_scale": 0
                        },
                        api_token
                    )
                    
                    result_url = output_urls[0]
                    
                    # --- 阶段 D: 展示与存入ZIP ---
                    img_data = download_image(result_url)
                    # 将图片写入内存中的ZIP
                    zip_file.writestr(f"AI_{img_file.name}", img_data)
                    generated_files_count += 1
                    
                    # 在界面上展示
                    with results_container:
                        c1, c2, c3 = st.columns([1, 1, 2])
                        c1.image(img_file, caption="原图", width=150)
                        c2.image(result_url, caption="AI生成", width=150)
                        with c3:
                            st.markdown(f"**原图内容识别:** `{content_desc}`")
                            st.markdown(f"**融合提示词:** `{final_combined_prompt[:100]}...`")
                        st.divider()

                except Exception as e:
                    st.error(f"处理 {img_file.name} 失败: {e}")
                
                # 更新进度
                progress_bar.progress((idx + 1) / len(batch_files))
                
                # --- 防封号等待机制 ---
                if idx < len(batch_files) - 1:
                    for i in range(5, 0, -1):
                        status_area.warning(f"☕ 冷却中 (避免接口拥堵): {i}s ...")
                        time.sleep(1)

        status_area.success("✅ 所有任务完成！")
        
        # --- 批量下载按钮 ---
        if generated_files_count > 0:
            st.markdown("### 📥 下载中心")
            st.download_button(
                label=f"📦 一键下载所有结果 (ZIP包)",
                data=zip_buffer.getvalue(),
                file_name="ai_generated_images.zip",
                mime="application/zip",
                use_container_width=True
            )

elif not api_token:
    st.info("👈 请在左侧输入 API Token 开始使用")
