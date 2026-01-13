import streamlit as st
import replicate
import time
import requests
import zipfile
import io
from PIL import Image
from replicate.exceptions import ReplicateError

# --- 页面基础设置 ---
st.set_page_config(page_title="AI风格重绘工作台 Pro", layout="wide")
st.title("🎨 AI风格重绘工作台 Pro (参数逻辑修复版)")
st.markdown("ℹ️ **修复说明**：已修正风格强度逻辑。现在调高滑块，画面会有巨大的风格变化。")

# --- 侧边栏：全局设置 ---
with st.sidebar:
    st.header("🔑 密钥设置")
    raw_token = st.text_input("Replicate API Token", type="password", help="r8_开头")
    api_token = raw_token.strip() if raw_token else None

    st.divider()
    
    st.header("🎮 风格控制中枢")
    # 【关键修复】调整了滑块的说明和默认值
    strength = st.slider(
        "风格重塑幅度 (Prompt Strength)", 
        0.0, 1.0, 0.75, 
        help="🔴 0.1-0.3: 几乎不变，只修细节\n🟡 0.4-0.6: 风格融合，保留轮廓\n🟢 0.7-0.9: 彻底转绘 (二次元转3D推荐选这里！)"
    )
    
    # 负面提示词
    default_neg = "anime, cartoon, drawing, sketch, 2d, illustration, flat, low quality, bad anatomy, blur, watermark, text, signature"
    negative_prompt = st.text_area("负面提示词 (禁止出现)", value=default_neg, height=100)
    
    st.info("💡 想要二次元转 3D，请将上面的滑块拉到 0.75 或 0.8，效果立竿见影。")

# --- 核心工具函数 ---

def preprocess_image(file_obj):
    """清洗图片格式，防止 tensor 错误"""
    try:
        image = Image.open(file_obj)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        return buf
    except Exception as e:
        raise Exception(f"图片格式清洗失败: {e}")

def run_replicate_dynamic(model_name, input_data, token):
    """自动获取最新版本并运行"""
    client = replicate.Client(api_token=token)
    
    try:
        model = client.models.get(model_name)
        latest_version = model.latest_version
    except Exception as e:
        raise Exception(f"模型 {model_name} 连接失败: {e}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            prediction = client.predictions.create(version=latest_version, input=input_data)
            prediction.wait()
            
            if prediction.status == "succeeded":
                return prediction.output
            elif prediction.status == "failed":
                if prediction.error and "NSFW" in str(prediction.error):
                    raise Exception("NSFW_ERROR")
                raise Exception(f"生成失败: {prediction.error}")
                
        except Exception as e:
            if str(e) == "NSFW_ERROR":
                raise e 
            err_str = str(e)
            if "429" in err_str or "throttled" in err_str:
                wait_time = 10 + (attempt * 5)
                st.toast(f"⏳ 限流冷却中... {wait_time}s", icon="🛡️")
                time.sleep(wait_time)
                continue
            raise e
            
    raise Exception("重试超时")

def download_image(url):
    response = requests.get(url)
    return response.content

# --- 布局：左右分栏 ---
left_col, right_col = st.columns([1, 1.5], gap="large")

# ================= 左侧：参考图 =================
with left_col:
    st.header("1️⃣ 参考图 (Style Source)")
    ref_file = st.file_uploader("上传参考图", type=['png', 'jpg', 'jpeg'], key="ref")
    
    style_tags = ""
    
    if ref_file:
        st.image(ref_file, use_container_width=True)
        
        if api_token:
            if st.button("🔍 分析参考图风格", type="primary"):
                with st.spinner("正在提取风格关键词..."):
                    try:
                        clean_ref = preprocess_image(ref_file)
                        output = run_replicate_dynamic(
                            "pharmapsychotic/clip-interrogator",
                            {"image": clean_ref, "mode": "fast"},
                            api_token
                        )
                        st.session_state['style_prompt'] = output
                    except Exception as e:
                        st.error(f"分析失败: {e}")

    if 'style_prompt' in st.session_state:
        st.markdown("##### 🎯 提取到的风格词:")
        style_prompt = st.text_area("风格提示词", value=st.session_state['style_prompt'], height=120)
    else:
        style_prompt = ""


# ================= 右侧：批量处理 =================
with right_col:
    st.header("2️⃣ 批量处理 (Content Source)")
    batch_files = st.file_uploader("批量上传图片", accept_multiple_files=True, key="batch")
    
    if 'batch_data' not in st.session_state:
        st.session_state['batch_data'] = {} 

    # --- 步骤 A: 识别内容 ---
    if batch_files and api_token:
        if st.button("👁️ 第一步：识别图片内容"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(batch_files):
                status_text.info(f"正在识别: {file.name}")
                try:
                    clean_file = preprocess_image(file)
                    content_desc = run_replicate_dynamic(
                        "salesforce/blip",
                        {"image": clean_file, "task": "image_captioning"},
                        api_token
                    )
                    # 清洗二次元相关词汇，防止干扰3D化
                    content_clean = content_desc.replace("cartoon", "").replace("anime", "").replace("drawing", "").strip()
                    
                    st.session_state['batch_data'][file.name] = {
                        "content": content_clean,
                        "status": "ready"
                    }
                except Exception as e:
                    st.error(f"{file.name} 识别失败: {e}")
                
                progress_bar.progress((i + 1) / len(batch_files))
            status_text.success("✅ 内容识别完成！")

        st.divider()

        # --- 步骤 B: 一键生成 ---
        if batch_files:
            if style_prompt:
                if st.button("🚀 第二步：一键生成 (应用风格)"):
                    if not st.session_state.get('batch_data'):
                        st.warning("请先点击第一步")
                    else:
                        zip_buffer = io.BytesIO()
                        has_results = False
                        progress = st.progress(0)
                        
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                            
                            for idx, file in enumerate(batch_files):
                                file_data = st.session_state['batch_data'].get(file.name, {})
                                content_txt = file_data.get("content", "content")
                                
                                # 强制增加 3D 关键词，强化效果
                                final_prompt = f"{style_prompt}, {content_txt}, 3d render, unreal engine 5, hyperrealistic, 8k, best quality"
                                
                                try:
                                    clean_input = preprocess_image(file)
                                    
                                    # 【关键修改】直接使用 strength，不再使用 1.0 - strength
                                    output = run_replicate_dynamic(
                                        "stability-ai/sdxl",
                                        {
                                            "image": clean_input,
                                            "prompt": final_prompt,
                                            "negative_prompt": negative_prompt,
                                            "prompt_strength": strength, # 这里改了！直接用滑块值
                                            "num_inference_steps": 40,   # 增加步数提高质量
                                            "guidance_scale": 7.5
                                        },
                                        api_token
                                    )
                                    
                                    img_url = output[0]
                                    img_bytes = download_image(img_url)
                                    zip_file.writestr(f"Styled_{file.name}", img_bytes)
                                    
                                    st.session_state['batch_data'][file.name]['result'] = img_url
                                    st.session_state['batch_data'][file.name]['final_prompt'] = final_prompt
                                    has_results = True
                                    
                                except Exception as e:
                                    err_msg = str(e)
                                    if "NSFW_ERROR" in err_msg:
                                        st.session_state['batch_data'][file.name]['error'] = "❌ 敏感内容跳过"
                                    else:
                                        st.session_state['batch_data'][file.name]['error'] = f"失败: {err_msg}"
                                
                                progress.progress((idx + 1) / len(batch_files))
                        
                        if has_results:
                            st.download_button(
                                "📦 批量下载 (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name="results.zip",
                                mime="application/zip",
                                type="primary"
                            )

            # --- 列表展示 ---
            st.write("### 🖼️ 结果预览")
            for file in batch_files:
                data = st.session_state['batch_data'].get(file.name, {})
                
                with st.expander(f"图片: {file.name}", expanded=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        st.image(file, caption="原图", width=150)
                    with c2:
                        current_content = data.get("content", "...")
                        preview_prompt = f"【风格】: {style_prompt[:50]}...\n【内容】: {current_content}"
                        st.text_area("提示词", value=preview_prompt, height=100, disabled=True, key=f"t_{file.name}")
                        if "error" in data: st.error(data["error"])
                    with c3:
                        if "result" in data:
                            st.image(data["result"], caption="结果", width=150)
                        else:
                            st.markdown("...")

if not api_token:
    st.warning("👈 请输入 Token")
