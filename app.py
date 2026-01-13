import streamlit as st
import replicate
import time
import requests
import zipfile
import io
from replicate.exceptions import ReplicateError

# --- 页面基础设置 ---
st.set_page_config(page_title="AI风格重绘工作台 Pro", layout="wide")
st.title("🎨 AI风格重绘工作台 Pro (二次元转3D/风格统一)")

# --- 侧边栏：全局设置 ---
with st.sidebar:
    st.header("🔑 密钥设置")
    raw_token = st.text_input("Replicate API Token", type="password", help="r8_开头")
    api_token = raw_token.strip() if raw_token else None

    st.divider()
    
    st.header("🎮 风格控制中枢")
    # 关键参数：重绘幅度
    strength = st.slider(
        "风格重塑幅度 (Prompt Strength)", 
        0.1, 1.0, 0.75, 
        help="核心参数！\n0.3-0.5: 微调，几乎不变\n0.6-0.8: 风格大变但保留构图 (推荐)\n0.9-1.0: 完全重画"
    )
    
    # 负面提示词：用于去除原图风格
    default_neg = "anime, cartoon, drawing, sketch, 2d, illustration, low quality, bad anatomy, blur"
    negative_prompt = st.text_area("负面提示词 (去除的元素)", value=default_neg, height=100, help="想把二次元转3D，这里务必加上 anime, 2d")
    
    st.info("💡 提示：如果生成的图变化不大，请调高【风格重塑幅度】到 0.8 以上。")

# --- 核心工具函数 ---
def run_replicate_dynamic(model_name, input_data, token):
    """自动获取最新版本并运行，带防限流和NSFW捕获"""
    client = replicate.Client(api_token=token)
    
    # 1. 动态获取最新版本
    try:
        model = client.models.get(model_name)
        latest_version = model.latest_version
    except Exception as e:
        raise Exception(f"模型 {model_name} 连接失败: {e}")

    # 2. 运行预测 (带重试)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prediction = client.predictions.create(version=latest_version, input=input_data)
            prediction.wait()
            
            if prediction.status == "succeeded":
                return prediction.output
            elif prediction.status == "failed":
                # 捕获 NSFW 错误
                if "NSFW" in str(prediction.error):
                    raise Exception("NSFW_ERROR")
                raise Exception(f"生成失败: {prediction.error}")
                
        except Exception as e:
            if str(e) == "NSFW_ERROR":
                raise e # 直接抛出给上层处理
            
            if "429" in str(e) or "throttled" in str(e):
                wait_time = 10 + (attempt * 5)
                st.toast(f"⏳ 限流保护中，冷却 {wait_time} 秒...", icon="🛡️")
                time.sleep(wait_time)
                continue
            raise e
            
    raise Exception("重试超时")

def download_image(url):
    response = requests.get(url)
    return response.content

# --- 布局：左右分栏 ---
left_col, right_col = st.columns([1, 1.5], gap="large")

# ================= 左侧：参考图 (风格源) =================
with left_col:
    st.header("1️⃣ 参考图 (Style Source)")
    st.caption("上传你想模仿的风格图片（如：游戏CG、电影剧照）")
    
    ref_file = st.file_uploader("上传参考图", type=['png', 'jpg', 'jpeg'], key="ref")
    
    style_tags = ""
    
    if ref_file:
        st.image(ref_file, use_container_width=True)
        
        if api_token:
            if st.button("🔍 分析参考图风格", type="primary"):
                with st.spinner("正在提取风格关键词..."):
                    try:
                        # 使用 CLIP Interrogator 提取风格
                        output = run_replicate_dynamic(
                            "pharmapsychotic/clip-interrogator",
                            {"image": ref_file, "mode": "fast"},
                            api_token
                        )
                        st.session_state['style_prompt'] = output
                    except Exception as e:
                        st.error(f"分析失败: {e}")

    # 风格提示词展示区
    if 'style_prompt' in st.session_state:
        st.markdown("##### 🎯 提取到的风格词:")
        style_prompt = st.text_area(
            "风格提示词 (会自动应用到右侧)", 
            value=st.session_state['style_prompt'], 
            height=120,
            key="style_input"
        )
    else:
        style_prompt = ""


# ================= 右侧：批量处理 (内容源) =================
with right_col:
    st.header("2️⃣ 批量处理 (Content Source)")
    st.caption("上传需要转绘的图片（如：二次元线稿、草图）")
    
    batch_files = st.file_uploader("批量上传图片", accept_multiple_files=True, key="batch")
    
    # 状态存储
    if 'batch_data' not in st.session_state:
        st.session_state['batch_data'] = {} # 用于存每张图的提示词和结果

    # --- 步骤 A: 批量识别内容 ---
    if batch_files and api_token:
        if st.button("👁️ 第一步：识别所有图片内容 (保留构图)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(batch_files):
                status_text.info(f"正在识别第 {i+1} 张: {file.name}...")
                try:
                    # 使用 BLIP 识别画面内容 (它通常只描述内容，不带风格)
                    content_desc = run_replicate_dynamic(
                        "salesforce/blip",
                        {"image": file, "task": "image_captioning"},
                        api_token
                    )
                    # 清洗内容描述，去掉 potential style words
                    content_clean = content_desc.replace("cartoon", "").replace("anime", "").strip()
                    
                    st.session_state['batch_data'][file.name] = {
                        "content": content_clean,
                        "status": "ready"
                    }
                except Exception as e:
                    st.error(f"{file.name} 识别失败: {e}")
                
                progress_bar.progress((i + 1) / len(batch_files))
            status_text.success("✅ 内容识别完成！请查看下方列表")

        st.divider()

        # --- 步骤 B: 列表展示与一键生成 ---
        if batch_files:
            # 只有当有风格词时才显示生成按钮
            if style_prompt:
                if st.button("🚀 第二步：一键统一风格并生成 (Style Transfer)"):
                    if not st.session_state.get('batch_data'):
                        st.warning("请先点击上方的【第一步：识别所有图片内容】")
                    else:
                        # 初始化下载包
                        zip_buffer = io.BytesIO()
                        has_results = False
                        
                        result_container = st.container()
                        progress = st.progress(0)
                        
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                            
                            for idx, file in enumerate(batch_files):
                                file_data = st.session_state['batch_data'].get(file.name, {})
                                content_txt = file_data.get("content", "original content")
                                
                                # === 核心逻辑：风格替换 ===
                                # 最终提示词 = 参考图风格 + 批量图内容
                                final_prompt = f"{style_prompt}, {content_txt}, best quality, 8k, masterpiece"
                                
                                try:
                                    # 调用 SDXL
                                    output = run_replicate_dynamic(
                                        "stability-ai/sdxl",
                                        {
                                            "image": file,
                                            "prompt": final_prompt,
                                            "negative_prompt": negative_prompt, # 强力去除原风格
                                            "prompt_strength": 1.0 - strength,  # 这里 Replicate 逻辑：0.2表示很像原图，0.8表示很像提示词
                                            "num_inference_steps": 30,
                                            "guidance_scale": 7.5
                                        },
                                        api_token
                                    )
                                    
                                    # 存结果
                                    img_url = output[0]
                                    img_bytes = download_image(img_url)
                                    zip_file.writestr(f"Styled_{file.name}", img_bytes)
                                    
                                    # 更新 session 状态用于展示
                                    st.session_state['batch_data'][file.name]['result'] = img_url
                                    st.session_state['batch_data'][file.name]['final_prompt'] = final_prompt
                                    has_results = True
                                    
                                except Exception as e:
                                    err_msg = str(e)
                                    if "NSFW" in err_msg:
                                        st.session_state['batch_data'][file.name]['error'] = "❌ 包含敏感内容 (NSFW)，已跳过"
                                    else:
                                        st.session_state['batch_data'][file.name]['error'] = f"生成失败: {err_msg}"
                                
                                progress.progress((idx + 1) / len(batch_files))
                        
                        if has_results:
                            st.download_button(
                                "📦 批量下载所有结果 (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name="style_transfer_results.zip",
                                mime="application/zip",
                                type="primary"
                            )

            # --- 列表展示区域 ---
            st.write("### 🖼️ 图片处理列表")
            for file in batch_files:
                data = st.session_state['batch_data'].get(file.name, {})
                
                with st.expander(f"图片: {file.name}", expanded=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    
                    # 第一列：原图
                    with c1:
                        st.image(file, caption="原图", width=150)
                    
                    # 第二列：提示词控制
                    with c2:
                        current_content = data.get("content", "等待识别...")
                        # 预览最终组合
                        preview_prompt = f"【风格】: {style_prompt[:50]}...\n【内容】: {current_content}"
                        st.text_area("当前图片提示词预览", value=preview_prompt, height=100, disabled=True)
                        
                        if "error" in data:
                            st.error(data["error"])
                    
                    # 第三列：结果图
                    with c3:
                        if "result" in data:
                            st.image(data["result"], caption="风格化结果", width=150)
                        else:
                            st.markdown("*等待生成...*")

elif not api_token:
    st.info("👈 请先在左侧输入 API Token")
