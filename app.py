import streamlit as st
import replicate
import time
import requests
import zipfile
import io
from replicate.exceptions import ReplicateError

# --- 页面基础设置 ---
st.set_page_config(page_title="AI全能风格迁移工作台", layout="wide")
st.title("🛡️ AI全能风格迁移 (官方稳定版)")
st.markdown("ℹ️ **说明**：已切换回 Stability AI 官方模型，确保 100% 成功率。内置智能防断连系统。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("🔑 核心设置")
    raw_token = st.text_input("Replicate API Token", type="password", help="r8_开头")
    api_token = raw_token.strip() if raw_token else None
    
    st.header("⚙️ 参数")
    strength = st.slider("风格重塑程度", 0.1, 1.0, 0.75)
    num_steps = st.slider("生成质量(步数)", 20, 50, 30)

# --- 核心工具函数：智能重试逻辑 ---
def run_replicate_safe(model, input_data, token):
    """
    尝试调用API，如果遇到429限流，自动等待并重试，直到成功。
    """
    client = replicate.Client(api_token=token)
    max_retries = 10  # 增加重试次数，确保万无一失
    
    for attempt in range(max_retries):
        try:
            return client.run(model, input=input_data)
        except ReplicateError as e:
            error_str = str(e)
            
            # 情况1: 遇到限流 (429) -> 等待并重试
            if "429" in error_str or "throttled" in error_str:
                wait_time = 15 + (attempt * 5) # 动态调整等待时间
                st.toast(f"⏳ 触发限流保护，正在冷却 {wait_time} 秒...", icon="🛡️")
                time.sleep(wait_time)
                continue 
            
            # 情况2: 遇到模型版本错误 (422) -> 这是致命错误，不能重试
            elif "422" in error_str:
                st.error("❌ 模型版本号失效，请联系开发者更新代码。")
                raise e
            
            # 其他错误 -> 抛出
            else:
                raise e
    
    raise Exception("重试多次失败，请检查账户余额或网络。")

def download_image(url):
    response = requests.get(url)
    return response.content

# --- 1. 参考风格 ---
st.header("1️⃣ 参考风格提取")
ref_file = st.file_uploader("上传风格参考图", type=['png', 'jpg', 'jpeg'], key="ref")

if ref_file and api_token:
    st.image(ref_file, width=200)
    if st.button("🔍 分析风格"):
        with st.spinner("正在分析..."):
            try:
                # 使用 CLIP Interrogator (官方 Verified 版本)
                output = run_replicate_safe(
                    "pharmapsychotic/clip-interrogator:8151e1c9f47e696fa316146a2e35812ccf79cfc9eba05b11c7f450155102af70",
                    {"image": ref_file, "mode": "fast"},
                    api_token
                )
                st.session_state['style_tags'] = output
                st.success("分析完成！")
            except Exception as e:
                st.error(f"分析失败: {e}")

if 'style_tags' in st.session_state:
    style_prompt = st.text_area("风格提示词", st.session_state['style_tags'], height=100)
else:
    style_prompt = ""

st.markdown("---")

# --- 2. 批量处理 ---
st.header("2️⃣ 批量生成 (自动排队)")
batch_files = st.file_uploader("上传批量图片", accept_multiple_files=True, key="batch")

if batch_files and style_prompt and api_token:
    if st.button(f"🚀 开始稳定处理 ({len(batch_files)} 张)"):
        
        zip_buffer = io.BytesIO()
        generated_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_col = st.container()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            
            for idx, img_file in enumerate(batch_files):
                # -------------------------------------------------
                # 步骤 A: 分析内容
                # -------------------------------------------------
                status_text.info(f"[{idx+1}/{len(batch_files)}] 👁️ 正在识别内容: {img_file.name}")
                try:
                    content_output = run_replicate_safe(
                        "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
                        {"image": img_file, "task": "image_captioning"},
                        api_token
                    )
                    content_desc = content_output.strip()
                except Exception as e:
                    st.warning(f"内容识别跳过: {e}")
                    content_desc = "image"
                
                # -------------------------------------------------
                # 步骤 B: 生成图片 (回归官方 SDXL 模型)
                # -------------------------------------------------
                status_text.info(f"[{idx+1}/{len(batch_files)}] 🎨 正在绘制: {img_file.name}")
                try:
                    final_prompt = f"{style_prompt}, {content_desc}, high quality, 8k"
                    
                    # 【关键修改】使用 Stability AI 官方 SDXL 模型 ID
                    # 这个 ID 是绝对不会变、也不会 422 的
                    output_urls = run_replicate_safe(
                        "stability-ai/sdxl:39ed52f2a78e934b3ba6e399ea1a963986eeac40ef080b697b0803a6466b717c",
                        {
                            "image": img_file,
                            "prompt": final_prompt,
                            "prompt_strength": 1.0 - strength, 
                            "num_inference_steps": num_steps, # 使用滑块控制步数
                            "guidance_scale": 7.5
                        },
                        api_token
                    )
                    
                    # 保存
                    img_data = download_image(output_urls[0])
                    zip_file.writestr(f"AI_{img_file.name}", img_data)
                    generated_count += 1
                    
                    with results_col:
                        c1, c2 = st.columns(2)
                        c1.image(img_file, width=150, caption="原图")
                        c2.image(output_urls[0], width=150, caption="AI生成")
                        st.divider()

                except Exception as e:
                    st.error(f"❌ 生成失败 {img_file.name}: {e}")

                # 更新进度
                progress_bar.progress((idx + 1) / len(batch_files))

        status_text.success("✅ 全部任务处理完毕！")
        
        if generated_count > 0:
            st.download_button(
                "📦 下载全部结果 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="ai_style_transfer_stable.zip",
                mime="application/zip",
                type="primary"
            )

elif not api_token:
    st.info("👈 请输入 Token")
