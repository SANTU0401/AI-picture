import streamlit as st
import replicate
import time
import requests
import zipfile
import io
from replicate.exceptions import ReplicateError

# --- 页面基础设置 ---
st.set_page_config(page_title="AI全能风格迁移工作台", layout="wide")
st.title("🛡️ AI全能风格迁移 (自动更新版)")
st.markdown("ℹ️ **说明**：系统现在会自动抓取 AI 模型的最新版本号，彻底解决版本过期 (422) 问题。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("🔑 核心设置")
    raw_token = st.text_input("Replicate API Token", type="password", help="r8_开头")
    api_token = raw_token.strip() if raw_token else None
    
    st.header("⚙️ 参数")
    strength = st.slider("风格重塑程度", 0.1, 1.0, 0.75)
    num_steps = st.slider("生成质量(步数)", 20, 50, 30)

# --- 核心工具函数：动态获取版本 + 智能重试 ---
def get_latest_version(model_name, token):
    """
    动态获取模型的最新版本ID，防止硬编码过期
    """
    client = replicate.Client(api_token=token)
    model = client.models.get(model_name)
    return model.latest_version

def run_replicate_dynamic(model_name, input_data, token):
    """
    自动查找最新版并运行，带防限流重试机制
    """
    client = replicate.Client(api_token=token)
    max_retries = 10
    
    # 第一步：获取最新版本 (只会执行一次，不消耗预测额度)
    try:
        latest_version = get_latest_version(model_name, token)
    except Exception as e:
        st.error(f"❌ 无法找到模型 {model_name}，可能是Token无效或模型被下架。")
        raise e

    # 第二步：执行预测 (带重试)
    for attempt in range(max_retries):
        try:
            # 使用 create 方法创建预测
            prediction = client.predictions.create(version=latest_version, input=input_data)
            
            # 等待结果
            prediction.wait()
            
            if prediction.status == "succeeded":
                return prediction.output
            else:
                raise Exception(f"生成失败，状态: {prediction.status}, 错误: {prediction.error}")

        except Exception as e:
            error_str = str(e)
            
            # 遇到限流 (429) -> 等待并重试
            if "429" in error_str or "throttled" in error_str:
                wait_time = 15 + (attempt * 5)
                st.toast(f"⏳ 触发限流保护，正在冷却 {wait_time} 秒...", icon="🛡️")
                time.sleep(wait_time)
                continue 
            
            # 其他错误 -> 抛出
            else:
                raise e
    
    raise Exception("重试多次失败，请检查账户余额。")

def download_image(url):
    response = requests.get(url)
    return response.content

# --- 1. 参考风格 ---
st.header("1️⃣ 参考风格提取")
ref_file = st.file_uploader("上传风格参考图", type=['png', 'jpg', 'jpeg'], key="ref")

if ref_file and api_token:
    st.image(ref_file, width=200)
    if st.button("🔍 分析风格"):
        with st.spinner("正在获取最新模型并分析..."):
            try:
                # 动态调用 CLIP Interrogator
                output = run_replicate_dynamic(
                    "pharmapsychotic/clip-interrogator", # 只写模型名，不写ID
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
    if st.button(f"🚀 开始处理 ({len(batch_files)} 张)"):
        
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
                    # 动态调用 BLIP
                    content_output = run_replicate_dynamic(
                        "salesforce/blip", # 只写模型名
                        {"image": img_file, "task": "image_captioning"},
                        api_token
                    )
                    content_desc = content_output.strip()
                except Exception as e:
                    st.warning(f"内容识别跳过: {e}")
                    content_desc = "image"
                
                # -------------------------------------------------
                # 步骤 B: 生成图片
                # -------------------------------------------------
                status_text.info(f"[{idx+1}/{len(batch_files)}] 🎨 正在绘制: {img_file.name}")
                try:
                    final_prompt = f"{style_prompt}, {content_desc}, high quality, 8k"
                    
                    # 动态调用 SDXL (使用官方 base 模型，最稳定)
                    output_urls = run_replicate_dynamic(
                        "stability-ai/sdxl", # 只写模型名，代码会自动找最新版ID
                        {
                            "image": img_file,
                            "prompt": final_prompt,
                            "prompt_strength": 1.0 - strength, 
                            "num_inference_steps": num_steps,
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
                file_name="ai_style_transfer_final.zip",
                mime="application/zip",
                type="primary"
            )

elif not api_token:
    st.info("👈 请输入 Token")

elif not api_token:
    st.info("👈 请输入 Token")
