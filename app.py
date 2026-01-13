import streamlit as st
import replicate
import time

# 页面配置
st.set_page_config(page_title="AI批量风格迁移工具", layout="wide")

st.title("🎨 AI图片风格提取与批量生成工具")
st.markdown("⚠️ **注意**：如果处理多张图片，请耐心等待，系统会自动排队以避免报错。")

# --- 侧边栏：设置 ---
with st.sidebar:
    st.header("🔑 设置")
    # 自动去空格
    raw_token = st.text_input("输入 Replicate API Token", type="password", help="以 r8_ 开头")
    api_token = raw_token.strip() if raw_token else None
    
    if api_token and not api_token.startswith("r8_"):
        st.error("⚠️ Token 格式看起来不对，必须以 r8_ 开头")

    st.header("⚙️ 生成参数")
    # 风格强度调整
    strength = st.slider("风格影响力 (Strength)", 0.1, 0.9, 0.7, help="值越大越像参考风格，值越小越像原图")
    num_steps = st.slider("生成步数", 20, 50, 30)

# --- 核心函数 ---
def run_replicate(model_version, input_data, token):
    client = replicate.Client(api_token=token)
    return client.run(model_version, input=input_data)

# --- 步骤 1: 上传与分析 ---
st.subheader("1. 上传参考风格图")
ref_file = st.file_uploader("上传参考图", type=['png', 'jpg', 'jpeg'])

if ref_file and api_token:
    st.image(ref_file, caption="参考图", width=250)
    
    if st.button("🔍 分析风格提示词"):
        with st.spinner("AI正在分析风格..."):
            try:
                # 使用 CLIP Interrogator 最新稳定版
                output = run_replicate(
                    "pharmapsychotic/clip-interrogator:8151e1c9f47e696fa316146a2e35812ccf79cfc9eba05b11c7f450155102af70",
                    {"image": ref_file, "mode": "fast"},
                    api_token
                )
                st.session_state['style_prompt'] = output
                st.success("✅ 风格提取成功！")
            except Exception as e:
                st.error(f"分析失败: {str(e)}")

# 显示提示词
if 'style_prompt' in st.session_state:
    st.markdown("### 📝 风格提示词")
    style_prompt = st.text_area("提示词 (可修改)", st.session_state['style_prompt'], height=80)
else:
    style_prompt = ""

st.markdown("---")

# --- 步骤 2: 批量处理 ---
st.subheader("2. 批量生成")
uploaded_files = st.file_uploader("上传内容图 (支持多选)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files and style_prompt and api_token:
    if st.button(f"🚀 开始生成 ({len(uploaded_files)} 张)"):
        
        progress_bar = st.progress(0)
        results_area = st.container()
        
        # 创建占位符，用于显示实时状态
        status_text = st.empty()
        
        for idx, img_file in enumerate(uploaded_files):
            # 更新进度条和文字
            progress_percent = (idx) / len(uploaded_files)
            progress_bar.progress(progress_percent)
            status_text.info(f"⏳ 正在处理第 {idx+1}/{len(uploaded_files)} 张图片: {img_file.name} ...")
            
            try:
                # 组合提示词
                final_prompt = f"{style_prompt}, high quality, 8k, detailed"
                
                # 【关键修改 1】使用 SDXL Base 1.0 的官方最新 Hash ID，修复 422 错误
                output = run_replicate(
                    "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02a319a1025b0004f2737463118197c",
                    {
                        "image": img_file,
                        "prompt": final_prompt,
                        "prompt_strength": 1.0 - strength,
                        "num_inference_steps": num_steps,
                        "guidance_scale": 7.5
                    },
                    api_token
                )
                
                # 展示结果
                with results_area:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img_file, caption=f"原图: {img_file.name}", width=200)
                    with col2:
                        img_url = output[0] if isinstance(output, list) else output
                        st.image(img_url, caption="AI生成图", width=200)
                        st.markdown(f"[下载大图]({img_url})")
                    st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ 图片 {img_file.name} 处理失败: {str(e)}")
            
            # 【关键修改 2】强制休息 12 秒，修复 429 限流错误
            if idx < len(uploaded_files) - 1: # 如果不是最后一张，就休息
                status_text.warning(f"☕ 为了防止限流报错，系统正在冷却 12 秒... (Replicate 限制)")
                time.sleep(12) 
            
        progress_bar.progress(1.0)
        status_text.success("✅ 所有图片处理完成！")

elif not api_token:
    st.warning("👈 请先在左侧输入 Token")
